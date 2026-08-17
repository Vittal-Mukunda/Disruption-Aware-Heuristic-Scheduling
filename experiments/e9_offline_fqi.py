"""E9 — offline value-learning baseline (fitted Q-iteration) vs DAHS.

Adds a faithful published-family competitor to the E2 comparison: an offline-RL
selector that learns Q(s, a) from logged round-robin transitions and deploys the
masked greedy policy (see `baselines/offline_fqi.py`). This closes the "no
reproduced competitor method" gap — the baselines so far are static rules, the
paper's own ablations, and generic algorithms (PPO, LinUCB).

Sub-commands (run in order):

    python -m experiments.e9_offline_fqi hpsearch         # fair HP search on held-out validation shifts
    python -m experiments.e9_offline_fqi eval             # train winner on 250 shifts, eval on 50 test shifts
    python -m experiments.e9_offline_fqi data_efficiency  # FQI sample-efficiency curve (25..250 shifts)
    python -m experiments.e9_offline_fqi summary          # comparison table, paired stats, figure

No frozen Phase 0-7 result is re-run or re-tuned. This only adds a new baseline
row; the DAHS numbers are read from the existing results/ parquets.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from baselines.offline_fqi import (
    MASKED_RULE,
    OfflineFQIPolicy,
    _load_regime_gmm,
    default_hp,
    load_offline_fqi,
    log_transitions,
    save_offline_fqi,
    subset_transitions,
    train_fqi,
)
from experiments.evaluate import canonical_test_seeds, evaluate_policy
from experiments.stats import require_metrics
from seed import shift_corpora
from simulation.heuristics import resolve_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "E9"
FIGURES_DIR = REPO_ROOT / "figures" / "E9"
RUN_DIR = REPO_ROOT / "runs" / "offline_fqi"
TRANSITIONS_CACHE = REPO_ROOT / "data" / "offline_fqi_transitions.npz"
HP_WINNER_PATH = RESULTS_DIR / "hp_winner.json"

N_HP_VALIDATION_SHIFTS = 50  # held-out validation slice of the 250 training shifts


def _train_seeds(cfg: DictConfig) -> list[int]:
    """The training block — the same shifts DAHS labels, so the comparison is paired."""
    return shift_corpora(cfg)["train"]


def _cache_stamp(cfg: DictConfig) -> np.ndarray:
    """Identity of the corpus a transition log was built from.

    Anything that changes what a transition MEANS goes in here: the action set
    and its order (action indices are positional), the observation width, the
    behaviour policy, and the objective weights that produced the rewards.
    """
    from simulation.state_extractor import N_FEATURES

    obj = cfg.objective
    return np.asarray([
        ",".join(resolve_pool(cfg)),
        str(N_FEATURES),
        str(cfg.labeling.observed_policy),
        f"{obj.w_breach},{obj.w_tardy},{obj.w_spoil},{obj.w_holding},"
        f"{obj.use_priority_weights}",
    ], dtype=object)


def _load_or_build_transitions(cfg: DictConfig) -> dict[str, np.ndarray]:
    """Transition log over the training block, cached to disk.

    THE CACHE IS STAMPED, because the alternative is a silent wrong answer. The
    behaviour policy is `cfg.labeling.observed_policy`, the same one that
    generated the DAHS state-visitation corpus — but the file used to carry no
    record of which pool, feature set or objective produced it, and was loaded
    blindly whenever it existed. Its own docstring said "delete the cache after
    changing it, the pool, or the objective", which makes correctness depend on
    someone remembering.

    That is not hypothetical. The committed cache holds 25-dimensional states and
    six-rule-era action indices from the pre-revision pool; loading it against the
    current 26-feature observation and the screened pool crashed in `train_fqi`
    with a bare `KeyError: 'n_actions'` — and a crash is the lucky outcome. Had
    the widths happened to agree, fitted Q-iteration would have trained happily on
    actions that meant different rules and the baseline would have been quietly
    wrong, which is the exact failure Stage 3 already guards against by refusing
    to train without a `label_meta.json`.

    A stale stamp rebuilds rather than raising: rebuilding is deterministic and
    bounded, and a baseline that silently skips itself is worse than one that
    costs a few minutes.
    """
    stamp = _cache_stamp(cfg)
    if TRANSITIONS_CACHE.exists():
        data = np.load(TRANSITIONS_CACHE, allow_pickle=True)
        cached = data["cache_stamp"] if "cache_stamp" in data.files else None
        if cached is not None and list(cached) == list(stamp):
            return {k: data[k] for k in data.files if k != "cache_stamp"}
        why = "no stamp (pre-revision cache)" if cached is None else (
            f"stamp mismatch\n     cached: {list(cached)}\n     current: {list(stamp)}"
        )
        print(f"[e9] discarding {TRANSITIONS_CACHE.name}: {why}")
    train_seeds = _train_seeds(cfg)
    print(f"[e9] logging transitions for {len(train_seeds)} shifts under "
          f"behaviour policy '{cfg.labeling.observed_policy}'...")
    t0 = time.perf_counter()
    transitions = log_transitions(train_seeds, cfg, resolve_pool(cfg))
    TRANSITIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(TRANSITIONS_CACHE, cache_stamp=stamp, **transitions)
    print(
        f"[e9] {transitions['states'].shape[0]} transitions "
        f"({time.perf_counter() - t0:.1f}s) -> {TRANSITIONS_CACHE.name}"
    )
    return transitions


def _hp_combos(cfg: DictConfig) -> list[dict]:
    grid = cfg.baselines.offline_fqi.hp_grid
    n_iter = int(cfg.baselines.offline_fqi.n_iterations)
    combos: list[dict] = []
    for gamma, depth, n_est, lr in itertools.product(
        list(grid.gamma), list(grid.max_depth),
        list(grid.n_estimators), list(grid.learning_rate),
    ):
        combos.append({
            "gamma": float(gamma),
            "n_iterations": n_iter,
            "xgb_params": {
                "max_depth": int(depth),
                "n_estimators": int(n_est),
                "learning_rate": float(lr),
            },
        })
    return combos


def _train_policy(
    transitions: dict[str, np.ndarray], hp: dict, cfg: DictConfig
) -> OfflineFQIPolicy:
    pool = resolve_pool(cfg)
    fefo_idx = pool.index(MASKED_RULE) if MASKED_RULE in pool else None
    model = train_fqi(
        transitions,
        gamma=hp["gamma"],
        n_iterations=hp["n_iterations"],
        xgb_params=hp["xgb_params"],
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        fefo_idx=fefo_idx,
        model_seed=int(cfg.seeds.model),
        verbose=False,
    )
    return OfflineFQIPolicy(
        model=model,
        fefo_threshold=float(cfg.heuristics.fefo_mask_threshold),
        arms=pool,
        regime_gmm=_load_regime_gmm(cfg),
    )


def _mean_kpis(df: pd.DataFrame) -> dict[str, float]:
    keys = ["composite_cost", "service_failure_rate", "mean_tardiness"]
    require_metrics(df, keys)
    return {k: float(df[k].mean()) for k in keys}


# --------------------------------------------------------------------------- #
# hpsearch                                                                     #
# --------------------------------------------------------------------------- #
def cmd_hpsearch(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    transitions = _load_or_build_transitions(cfg)
    train_seeds = _train_seeds(cfg)

    n_fit = len(_train_seeds(cfg)) - N_HP_VALIDATION_SHIFTS
    fit_ids = np.arange(n_fit)
    val_seeds = train_seeds[n_fit:]
    fit_transitions = subset_transitions(transitions, fit_ids)
    print(
        f"[e9 hpsearch] fit on shifts 0..{n_fit - 1} "
        f"({fit_transitions['states'].shape[0]} transitions); "
        f"select on validation shifts {n_fit}..{len(_train_seeds(cfg)) - 1}"
    )

    combos = _hp_combos(cfg)
    rows: list[dict] = []
    for i, hp in enumerate(combos):
        t0 = time.perf_counter()
        policy = _train_policy(fit_transitions, hp, cfg)
        df = evaluate_policy(
            "offline_fqi_hp", policy, val_seeds, cfg, save=False, verbose=False
        )
        kpis = _mean_kpis(df)
        rows.append({
            "gamma": hp["gamma"],
            "max_depth": hp["xgb_params"]["max_depth"],
            "n_estimators": hp["xgb_params"]["n_estimators"],
            "learning_rate": hp["xgb_params"]["learning_rate"],
            "val_service_failure_rate": kpis["service_failure_rate"],
            "val_composite_cost": kpis["composite_cost"],
        })
        print(
            f"  [{i + 1:>2d}/{len(combos)}] gamma={hp['gamma']:.2f} "
            f"depth={hp['xgb_params']['max_depth']} "
            f"n_est={hp['xgb_params']['n_estimators']} "
            f"lr={hp['xgb_params']['learning_rate']:.2f}  "
            f"val_cost={kpis['composite_cost']:.3f} "
            f"val_breach={kpis['service_failure_rate']:.4f} "
            f"({time.perf_counter() - t0:.0f}s)"
        )

    table = pd.DataFrame(rows).sort_values("val_composite_cost").reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(RESULTS_DIR / "hpsearch.parquet", index=False)

    best = table.iloc[0]
    winner = {
        "gamma": float(best["gamma"]),
        "n_iterations": int(cfg.baselines.offline_fqi.n_iterations),
        "xgb_params": {
            "max_depth": int(best["max_depth"]),
            "n_estimators": int(best["n_estimators"]),
            "learning_rate": float(best["learning_rate"]),
        },
    }
    with open(HP_WINNER_PATH, "w", encoding="utf-8") as fh:
        json.dump(winner, fh, indent=2)

    print(f"\n[e9 hpsearch] selected on validation composite_cost (lower is better):")
    print(table.to_string(index=False))
    print(f"\n[e9 hpsearch] winner -> {HP_WINNER_PATH.name}: {winner}")
    return 0


def _load_winner_hp(cfg: DictConfig) -> dict:
    if HP_WINNER_PATH.exists():
        with open(HP_WINNER_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    print("[e9] no hp_winner.json found — falling back to config defaults.")
    return default_hp(cfg)


# --------------------------------------------------------------------------- #
# eval                                                                         #
# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    transitions = _load_or_build_transitions(cfg)
    hp = _load_winner_hp(cfg)
    test_seeds = canonical_test_seeds(cfg)

    print(f"[e9 eval] training FQI on all {len(_train_seeds(cfg))} "
          f"shifts with hp={hp}")
    t0 = time.perf_counter()
    policy = _train_policy(transitions, hp, cfg)
    save_offline_fqi(RUN_DIR, policy.model, {
        "gamma": hp["gamma"],
        "n_iterations": hp["n_iterations"],
        "xgb_params": hp["xgb_params"],
        "fefo_threshold": float(cfg.heuristics.fefo_mask_threshold),
        "n_transitions": int(transitions["states"].shape[0]),
        # Needed by `load_offline_fqi` to rebuild the same action set and the
        # same feature vector at deployment.
        "arms": list(resolve_pool(cfg)),
        "behaviour_policy": str(cfg.labeling.observed_policy),
        "use_regime_features": bool(
            cfg.baselines.offline_fqi.get("use_regime_features", False)
        ),
        "state_dim": int(transitions["states"].shape[1]),
    })
    print(f"[e9 eval] trained + saved to {RUN_DIR} ({time.perf_counter() - t0:.0f}s)")

    df = evaluate_policy(
        "offline_fqi", policy, test_seeds, cfg,
        results_dir=None, save=True, verbose=args.verbose,
    )
    kpis = _mean_kpis(df)
    print(f"\n[e9 eval] offline_fqi over {len(df)} test shifts:")
    print(f"  service_failure_rate = {kpis['service_failure_rate']:.4f}")
    print(f"  composite_cost       = {kpis['composite_cost']:.4f}")
    print(f"  mean_tardiness  = {kpis['mean_tardiness']:.4f}")

    print("\n[e9 eval] context (frozen results/ parquets):")
    for m in ("ours", "snapshot_xgb", "greedy_mpc"):
        p = REPO_ROOT / "results" / f"{m}.parquet"
        if p.exists():
            k = _mean_kpis(pd.read_parquet(p))
            print(f"  {m:<14s} sla_breach={k['service_failure_rate']:.4f}  "
                  f"composite_cost={k['composite_cost']:.4f}  "
                  f"mean_tardiness={k['mean_tardiness']:.4f}")
    return 0


# --------------------------------------------------------------------------- #
# data_efficiency                                                               #
# --------------------------------------------------------------------------- #
def cmd_data_efficiency(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    transitions = _load_or_build_transitions(cfg)
    hp = _load_winner_hp(cfg)
    test_seeds = canonical_test_seeds(cfg)
    n_train = len(_train_seeds(cfg))

    budgets = [int(b) for b in cfg.experiments.e2.data_efficiency_budgets]
    n_reps = int(cfg.experiments.e2.data_efficiency_reps)

    rows: list[dict] = []
    for budget in budgets:
        reps = 1 if budget >= n_train else n_reps
        for rep in range(reps):
            if budget >= n_train:
                ids = np.arange(n_train)
            else:
                rng = np.random.default_rng(1000 * budget + rep)
                ids = rng.choice(n_train, size=budget, replace=False)
            sub = subset_transitions(transitions, ids)
            t0 = time.perf_counter()
            policy = _train_policy(sub, hp, cfg)
            df = evaluate_policy(
                "offline_fqi_de", policy, test_seeds, cfg, save=False, verbose=False
            )
            kpis = _mean_kpis(df)
            rows.append({
                "budget": budget, "rep": rep,
                "service_failure_rate": kpis["service_failure_rate"],
                "composite_cost": kpis["composite_cost"],
            })
            print(
                f"  budget={budget:>3d} rep={rep}  "
                f"sla_breach={kpis['service_failure_rate']:.4f} "
                f"composite_cost={kpis['composite_cost']:.3f} "
                f"({time.perf_counter() - t0:.0f}s)"
            )

    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "data_efficiency_offline_fqi.parquet"
    table.to_parquet(out, index=False)

    agg = table.groupby("budget").agg(
        service_failure_mean=("service_failure_rate", "mean"),
        service_failure_std=("service_failure_rate", "std"),
        composite_cost_mean=("composite_cost", "mean"),
        composite_cost_std=("composite_cost", "std"),
    ).reset_index()
    print(f"\n[e9 data_efficiency] offline_fqi by budget -> {out.name}:")
    print(agg.to_string(index=False))
    return 0


# --------------------------------------------------------------------------- #
# robustness_grid                                                                #
# --------------------------------------------------------------------------- #
def cmd_robustness_grid(args: argparse.Namespace) -> int:
    """Evaluate the frozen offline_fqi model across the 12-cell E8 untuned grid.

    Reuses the E8 grid definition and config-patching verbatim, so the cells are
    identical to the frozen DAHS / snapshot / greedy_mpc grid. No re-training.
    """
    from experiments.e8_robustness_grid import (
        ARRIVAL_RATES,
        SLA_TIGHTNESS_LEVELS,
        _build_robustness_cfg,
        _config_cell_id,
    )
    from experiments.stats import bootstrap_mean_ci

    cfg = OmegaConf.load(CONFIG_PATH)
    policy = load_offline_fqi(RUN_DIR, cfg=cfg)
    seeds = canonical_test_seeds(cfg)
    grid_dir = RESULTS_DIR / "robustness_grid"

    rows: list[dict] = []
    for arrival_rate in ARRIVAL_RATES:
        for sla_key, sla_tri in SLA_TIGHTNESS_LEVELS.items():
            cell_id = _config_cell_id(arrival_rate, sla_key)
            cell_cfg = _build_robustness_cfg(cfg, arrival_rate, sla_tri)
            df = evaluate_policy(
                "offline_fqi", policy, seeds, cell_cfg,
                results_dir=grid_dir / cell_id, save=True, verbose=False,
            )
            ci = bootstrap_mean_ci(
                df["service_failure_rate"].to_numpy(dtype=np.float64),
                n_resamples=10000, seed=1337,
            ).as_row()
            rows.append({
                "arrival_rate": float(arrival_rate), "sla_tightness": sla_key,
                "cell_id": cell_id,
                "service_failure_rate": float(df["service_failure_rate"].mean()),
                "sla_breach_ci_lo": float(ci["ci_lo"]),
                "sla_breach_ci_hi": float(ci["ci_hi"]),
                "composite_cost": float(df["composite_cost"].mean()),
            })
            print(f"  {cell_id:<20} sla_breach={rows[-1]['service_failure_rate']:.4f}  "
                  f"composite_cost={rows[-1]['composite_cost']:.3f}")

    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "robustness_grid_offline_fqi.parquet"
    table.to_parquet(out, index=False)

    # Side-by-side with the frozen DAHS grid (E8 summary).
    e8 = pd.read_parquet(
        REPO_ROOT / "results" / "E8" / "robustness_grid_summary.parquet"
    )
    dahs = e8[(e8["method"] == "ours") & (e8["metric"] == "service_failure_rate")]
    print(f"\n[e9 robustness_grid] offline_fqi vs DAHS — SLA breach, 12 untuned cells:")
    print(f"  {'cell':<20}{'offline_fqi':>13}{'DAHS':>11}{'winner':>13}")
    dahs_wins = 0
    for r in rows:
        m = dahs[(dahs["arrival_rate"] == r["arrival_rate"])
                 & (dahs["sla_tightness"] == r["sla_tightness"])]
        d = float(m["point"].iloc[0]) if not m.empty else float("nan")
        dahs_better = d <= r["service_failure_rate"]
        dahs_wins += int(dahs_better)
        print(f"  {r['cell_id']:<20}{r['service_failure_rate']:>13.4f}{d:>11.4f}"
              f"{'DAHS' if dahs_better else 'offline_fqi':>13}")
    print(f"\n[e9 robustness_grid] DAHS <= offline_fqi in {dahs_wins}/{len(rows)} "
          f"cells -> {out.name}")
    return 0


# --------------------------------------------------------------------------- #
# summary                                                                       #
# --------------------------------------------------------------------------- #
def _paired_bootstrap(
    diff: np.ndarray, n_resamples: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    """Mean and 95% bootstrap CI of a per-shift paired difference."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([
        diff[rng.integers(0, n, n)].mean() for _ in range(n_resamples)
    ])
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


def cmd_summary(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    results = REPO_ROOT / "results"

    fqi_path = results / "offline_fqi.parquet"
    if not fqi_path.exists():
        print("[e9 summary] results/offline_fqi.parquet missing — run `eval` first.")
        return 1
    fqi = pd.read_parquet(fqi_path)

    print("[e9 summary] main comparison (means over 50 test shifts):")
    methods = ["ours", "offline_fqi", "greedy_mpc", "snapshot_xgb"]
    frames: dict[str, pd.DataFrame] = {}
    for m in methods:
        p = results / f"{m}.parquet"
        if p.exists():
            frames[m] = pd.read_parquet(p)
            k = _mean_kpis(frames[m])
            print(f"  {m:<14s} sla_breach={k['service_failure_rate']:.4f}  "
                  f"composite_cost={k['composite_cost']:.4f}  "
                  f"mean_tardiness={k['mean_tardiness']:.4f}")

    # Paired DAHS - offline_fqi (negative => DAHS better; lower is better).
    if "ours" in frames:
        ours = frames["ours"].sort_values("shift_seed").reset_index(drop=True)
        fqi_s = fqi.sort_values("shift_seed").reset_index(drop=True)
        if list(ours["shift_seed"]) == list(fqi_s["shift_seed"]):
            print("\n[e9 summary] paired DAHS - offline_fqi "
                  "(negative => DAHS better; 95% bootstrap CI):")
            for metric in ("service_failure_rate", "composite_cost", "mean_tardiness"):
                diff = (ours[metric] - fqi_s[metric]).to_numpy()
                m, lo, hi = _paired_bootstrap(diff)
                verdict = "DAHS better" if hi < 0 else (
                    "offline_fqi better" if lo > 0 else "CI spans 0 (tie)")
                print(f"  {metric:<18s} mean={m:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]"
                      f"  -> {verdict}")
        else:
            print("\n[e9 summary] WARNING: shift_seed mismatch — skipping paired test.")

    # Data-efficiency figure (offline_fqi; overlay DAHS if its curve is on disk).
    de_path = RESULTS_DIR / "data_efficiency_offline_fqi.parquet"
    if de_path.exists():
        _plot_data_efficiency(pd.read_parquet(de_path))
    else:
        print("\n[e9 summary] no data_efficiency parquet — run `data_efficiency`.")
    return 0


def _load_dahs_data_efficiency() -> pd.DataFrame | None:
    """Load the frozen DAHS data-efficiency curve (5 budgets x 5 reps).

    Schema (per record): budget, rep, service_failure_rate_mean, composite_cost_mean.
    """
    p = REPO_ROOT / "results" / "data_efficiency" / "data_efficiency_summary.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return pd.DataFrame(json.loads(fh.read()))


def _plot_data_efficiency(de: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = de.groupby("budget").agg(
        mean=("service_failure_rate", "mean"), std=("service_failure_rate", "std")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    dahs = _load_dahs_data_efficiency()
    if dahs is not None:
        d = dahs.groupby("budget")["service_failure_rate_mean"].agg(
            ["mean", "std"]).reset_index()
        ax.errorbar(
            d["budget"], d["mean"], yerr=d["std"].fillna(0.0),
            marker="s", capsize=4, lw=2, color="steelblue",
            label="DAHS (ours) -- rollout distillation",
        )
    ax.errorbar(
        agg["budget"], agg["mean"], yerr=agg["std"].fillna(0.0),
        marker="o", capsize=4, lw=2, color="firebrick",
        label="offline_fqi -- fitted Q-iteration",
    )
    ax.set_xlabel("Training shifts")
    ax.set_ylabel("SLA-breach rate (50 test shifts)")
    ax.set_title("Sample efficiency: DAHS vs offline value-learning baseline")
    ax.grid(True, alpha=0.3)
    ax.legend()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"data_efficiency_offline_fqi.{ext}", dpi=120)
    plt.close(fig)
    print(f"[e9 summary] wrote {FIGURES_DIR / 'data_efficiency_offline_fqi.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="E9 — offline-FQI baseline vs DAHS.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_hp = sub.add_parser("hpsearch", help="fair HP search on validation shifts")
    p_hp.set_defaults(func=cmd_hpsearch)
    p_ev = sub.add_parser("eval", help="train winner on 250 shifts, eval on test")
    p_ev.add_argument("--verbose", action="store_true")
    p_ev.set_defaults(func=cmd_eval)
    p_de = sub.add_parser("data_efficiency", help="FQI sample-efficiency curve")
    p_de.set_defaults(func=cmd_data_efficiency)
    p_rg = sub.add_parser("robustness_grid",
                          help="evaluate frozen offline_fqi across the 12-cell grid")
    p_rg.set_defaults(func=cmd_robustness_grid)
    p_su = sub.add_parser("summary", help="comparison table, paired stats, figure")
    p_su.set_defaults(func=cmd_summary)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
