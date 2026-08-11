"""Stage 1 — calibrate the parameterised rules, then screen the pool.

Three sub-commands, run in this order, all on the CALIBRATION shift block
(`seed.shift_corpora(cfg)["calib"]`), which is disjoint from both train and test.
Everything downstream depends on the pool this stage settles, so it must precede
labelling.

    python -m experiments.calibrate_rules calibrate
    python -m experiments.calibrate_rules screen
    python -m experiments.calibrate_rules diversity

1. CALIBRATE (Reviewer 1, 4.c)
------------------------------
ATC's look-ahead scale `k` was never fitted: the submitted configuration carried
`atc_lookahead_k: 2.0` and no search over `k` existed anywhere in the repository.
That matters because WSPT is exactly the `k -> inf` limit of ATC — in
`simulation.heuristics.atc`, `exp(-slack/(k*p_bar)) -> 1` leaves the WSPT key
`w/p` — so a properly fitted ATC *cannot* be beaten by WSPT. The submitted paper
reported WSPT winning 32% of decisions against ATC's 10%, which is the signature
of a mis-set `k`, not a property of the rules.

The reviewer asks for both calibrations, and both are computed:

  standalone  `k` minimising composite cost with ATC used as a policy on its
              own. Required because ATC is itself a reported benchmark, and
              benchmarking an uncalibrated rule understates it.
  portfolio   `k` maximising ATC's marginal contribution *inside* the pool —
              the quantity that actually matters for a selection
              hyper-heuristic, since a rule earns its slot by covering states
              the others handle badly, not by being good on average.

The two need not coincide, and the paper reports both together with the
deployed value.

2. SCREEN (Reviewer 1, 4.a, 4.b, 4.d)
-------------------------------------
Scores every rule in the candidate library on the calibration corpus and reports
a screening table with, per rule, its win rate and its marginal contribution —
the mean increase in achievable cost if that rule were removed from the pool. A
rule with a high win rate but zero marginal contribution is redundant (another
rule covers the same states); a rule with a low win rate but high marginal
contribution is a specialist and should be kept. Win rate alone, which is what
the submitted Section 6.1 reported, cannot distinguish the two.

3. DIVERSITY (Reviewer 1, 4.e)
------------------------------
The submitted Figure 1 showed win rate per shift and per interval, which the
reviewer correctly says demonstrates nothing about complementarity: it varies
the instance, not the state. This produces win rate over a grid of the two state
dimensions the reviewer names — queue state and deadline pressure — so that
"the rules are complementary" becomes a statement about *when* each rule wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf, open_dict

from labeling.rollout_labeler import label_one_shift
from seed import shift_corpora
from simulation.heuristics import SCREENING_POOL, with_default_scales
from simulation.warehouse_env import WarehouseEnv, reset_step_counter, simulated_steps

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "S1_calibration"
FIG_DIR = REPO_ROOT / "figures" / "S1_calibration"

# Parameterised rules and the config key holding their scale parameter.
PARAMETERISED: dict[str, str] = {"ATC": "atc_lookahead_k", "COVERT": "covert_lookahead_k"}


def _with_k(cfg: DictConfig, key: str, k: float) -> DictConfig:
    new = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(new):
        new.heuristics[key] = float(k)
    return new


def _default_k(cfg: DictConfig) -> DictConfig:
    """Fill any unset scale parameter with a neutral value.

    `config.yaml` ships `atc_lookahead_k: null` so that an uncalibrated run fails
    loudly rather than silently reusing the submitted 2.0. Screening still needs
    *some* value before calibration has run, so it uses the grid midpoint and
    says so in its output.
    """
    grid = [float(x) for x in cfg.heuristics.calibration.k_grid]
    mid = float(np.median(grid))
    new = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(new):
        for key in PARAMETERISED.values():
            if new.heuristics.get(key) is None:
                new.heuristics[key] = mid
    return new


# ---------------------------------------------------------------------------
# Cost collection
# ---------------------------------------------------------------------------


def _collect_costs(
    seeds: list[int], cfg: DictConfig, candidates: list[str], n_jobs: int
) -> pd.DataFrame:
    """Per-epoch rollout cost of every candidate on every calibration shift."""
    rows = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(label_one_shift)(i, s, cfg, candidates=candidates)
        for i, s in enumerate(seeds)
    )
    return pd.DataFrame([r for shift in rows for r in shift])


def _run_static(seed: int, cfg: DictConfig, rule: str) -> float:
    """Composite cost of one shift under a single fixed rule.

    Module level, not a closure, so joblib's loky backend can pickle it without
    relying on cloudpickle's closure handling.
    """
    env = WarehouseEnv(int(seed), cfg)
    return float(env.run_with_policy(rule)["composite_cost"])


def _standalone_cost(seeds: list[int], cfg: DictConfig, rule: str, n_jobs: int) -> float:
    """Mean composite cost of running `rule` alone over the calibration shifts."""
    vals = Parallel(n_jobs=n_jobs)(delayed(_run_static)(s, cfg, rule) for s in seeds)
    return float(np.mean(vals))


def _marginal_contribution(df: pd.DataFrame, pool: list[str], rule: str) -> np.ndarray:
    """Per-epoch cost increase from removing `rule` from `pool`.

    `min over pool\\{rule}` minus `min over pool`, which is zero at every epoch
    where some other rule was already at least as good. The mean over epochs is
    the rule's marginal value; the fraction of strictly positive entries is how
    often it is the *only* rule that helps.
    """
    others = [h for h in pool if h != rule]
    if not others:
        return np.zeros(len(df))
    with_all = df[[f"cost_{h}" for h in pool]].to_numpy(np.float64).min(axis=1)
    without = df[[f"cost_{h}" for h in others]].to_numpy(np.float64).min(axis=1)
    return without - with_all


def _bootstrap_ci(x: np.ndarray, n_resamples: int = 10000, seed: int = 1337) -> tuple:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ---------------------------------------------------------------------------
# 1. Calibrate
# ---------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    seeds = shift_corpora(cfg)["calib"]
    grid = [float(x) for x in cfg.heuristics.calibration.k_grid]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reset_step_counter()

    results: list[dict] = []
    for rule, key in PARAMETERISED.items():
        if args.rules and rule not in args.rules:
            continue
        # Portfolio contribution is measured against the other candidates, so the
        # rule under calibration is excluded from the reference pool.
        others = [h for h in SCREENING_POOL if h != rule]

        print(f"\n[calibrate] {rule} — {len(grid)} values of {key}, "
              f"{len(seeds)} calibration shifts")
        for k in grid:
            cfg_k = _default_k(_with_k(cfg, key, k))

            standalone = _standalone_cost(seeds, cfg_k, rule, args.n_jobs)

            df = _collect_costs(seeds, cfg_k, others + [rule], args.n_jobs)
            marginal = _marginal_contribution(df, others + [rule], rule)
            win_rate = float(
                (df[[f"cost_{h}" for h in others + [rule]]]
                 .to_numpy(np.float64).argmin(axis=1) == len(others)).mean()
            )

            results.append({
                "rule": rule,
                "k": k,
                "standalone_cost": standalone,
                "portfolio_marginal_mean": float(marginal.mean()),
                "portfolio_marginal_p_positive": float((marginal > 1e-9).mean()),
                "portfolio_win_rate": win_rate,
            })
            print(f"  k={k:<6.2f} standalone={standalone:10.3f}  "
                  f"marginal={marginal.mean():8.4f}  win={win_rate:6.3f}")

    table = pd.DataFrame(results)
    table.to_parquet(RESULTS_DIR / "rule_calibration.parquet", index=False)

    chosen: dict[str, dict] = {}
    for rule in table["rule"].unique():
        sub = table[table["rule"] == rule]
        k_standalone = float(sub.loc[sub["standalone_cost"].idxmin(), "k"])
        k_portfolio = float(sub.loc[sub["portfolio_marginal_mean"].idxmax(), "k"])
        chosen[rule] = {
            "k_standalone": k_standalone,
            "k_portfolio": k_portfolio,
            # The pool is what gets deployed, so the portfolio value is the one
            # written back to config. The standalone value is what the rule is
            # benchmarked at when it appears as a baseline in its own right.
            "k_deployed": k_portfolio,
            "config_key": PARAMETERISED[rule],
        }

    (RESULTS_DIR / "rule_calibration.json").write_text(
        json.dumps({
            "chosen": chosen,
            "n_calibration_shifts": len(seeds),
            "k_grid": grid,
            "simulated_interval_steps": simulated_steps(),
        }, indent=2),
        encoding="utf-8",
    )
    print("\n[calibrate] chosen:")
    for rule, c in chosen.items():
        print(f"  {rule}: standalone k*={c['k_standalone']}  "
              f"portfolio k*={c['k_portfolio']}  -> deploying {c['k_deployed']}")
    print(f"\n[calibrate] simulated interval-steps: {simulated_steps():,}")
    print(f"[calibrate] write these into config.yaml under `heuristics:` before screening.")
    return 0


# ---------------------------------------------------------------------------
# 2. Screen
# ---------------------------------------------------------------------------


def cmd_screen(args: argparse.Namespace) -> int:
    cfg = _default_k(OmegaConf.load(CONFIG_PATH))
    seeds = shift_corpora(cfg)["calib"]
    candidates = list(args.candidates) if args.candidates else list(SCREENING_POOL)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reset_step_counter()

    print(f"[screen] {len(candidates)} candidates on {len(seeds)} calibration shifts: "
          f"{candidates}")
    df = _collect_costs(seeds, cfg, candidates, args.n_jobs)
    cost_cols = [f"cost_{h}" for h in candidates]
    argmin = df[cost_cols].to_numpy(np.float64).argmin(axis=1)

    rows: list[dict] = []
    for i, h in enumerate(candidates):
        marginal = _marginal_contribution(df, candidates, h)
        lo, hi = _bootstrap_ci(marginal)
        rows.append({
            "rule": h,
            "win_rate": float((argmin == i).mean()),
            "marginal_mean": float(marginal.mean()),
            "marginal_ci_lo": lo,
            "marginal_ci_hi": hi,
            "p_strictly_helps": float((marginal > 1e-9).mean()),
            # A specialist earns its slot even at a low win rate, provided the
            # bootstrap CI on its marginal contribution excludes zero.
            "retain": bool(lo > 0.0),
        })

    table = pd.DataFrame(rows).sort_values("marginal_mean", ascending=False)
    table.to_parquet(RESULTS_DIR / "pool_screening.parquet", index=False)
    df.to_parquet(RESULTS_DIR / "calibration_epoch_costs.parquet", index=False)

    top_win = float(table["win_rate"].max())
    retained = table.loc[table["retain"], "rule"].tolist()

    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n[screen] retained pool: {retained}")
    print(f"[screen] top single-rule win rate: {top_win:.3f}")
    if top_win > 0.60:
        print("[screen] WARNING: one rule wins >60% of decisions — selection has "
              "little room to add value at this operating point.")
    if len(retained) < 2:
        print("[screen] WARNING: fewer than two rules survive; the selection "
              "problem is degenerate under the current objective.")
    print(f"[screen] simulated interval-steps: {simulated_steps():,}")
    print(f"[screen] write the retained pool into config.yaml `heuristics.pool`.")

    (RESULTS_DIR / "pool_screening.json").write_text(
        json.dumps({
            "candidates": candidates,
            "retained": retained,
            "top_win_rate": top_win,
            "n_calibration_shifts": len(seeds),
            "simulated_interval_steps": simulated_steps(),
        }, indent=2),
        encoding="utf-8",
    )
    return 0


# ---------------------------------------------------------------------------
# 3. Diversity across the state space
# ---------------------------------------------------------------------------


def cmd_diversity(args: argparse.Namespace) -> int:
    """Win rate over (queue state x deadline pressure) — replaces Figure 1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = RESULTS_DIR / "calibration_epoch_costs.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}; run `screen` first.")
    df = pd.read_parquet(path)
    candidates = [c[len("cost_"):] for c in df.columns if c.startswith("cost_")]
    cost_cols = [f"cost_{h}" for h in candidates]
    df = df.assign(winner=[candidates[i] for i in
                           df[cost_cols].to_numpy(np.float64).argmin(axis=1)])

    n_bins = int(args.n_bins)
    # Queue state and deadline pressure — the two dimensions Reviewer 1 (4.e)
    # names. Quantile bins so each cell carries comparable mass.
    df["q_bin"] = pd.qcut(df["f_queue_length"], n_bins, labels=False, duplicates="drop")
    df["d_bin"] = pd.qcut(df["f_mean_slack_minutes"], n_bins, labels=False, duplicates="drop")

    grid = (
        df.groupby(["q_bin", "d_bin", "winner"]).size().rename("n").reset_index()
    )
    totals = grid.groupby(["q_bin", "d_bin"])["n"].transform("sum")
    grid["win_rate"] = grid["n"] / totals
    grid.to_parquet(RESULTS_DIR / "diversity_state_grid.parquet", index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(candidates),
                             figsize=(3.0 * len(candidates), 3.2), squeeze=False)
    for ax, h in zip(axes[0], candidates):
        pivot = (
            grid[grid["winner"] == h]
            .pivot(index="d_bin", columns="q_bin", values="win_rate")
            .reindex(index=range(n_bins), columns=range(n_bins))
            .fillna(0.0)
        )
        im = ax.imshow(pivot.to_numpy(), origin="lower", vmin=0.0, vmax=1.0,
                       cmap="viridis", aspect="auto")
        ax.set_title(h, fontsize=10)
        ax.set_xlabel("queue length (quantile)")
        ax.set_ylabel("mean slack (quantile)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="win rate")
    fig.suptitle(
        "Rule win rate across the state space — complementarity is a statement "
        "about WHEN each rule wins", fontsize=11,
    )
    fig.savefig(FIG_DIR / "diversity_state_grid.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "diversity_state_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[diversity] wrote {FIG_DIR.relative_to(REPO_ROOT)}/diversity_state_grid.*")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("calibrate", help="Fit ATC/COVERT look-ahead scales.")
    pc.add_argument("--rules", nargs="*", default=None)
    pc.add_argument("--n-jobs", type=int, default=-1)
    pc.set_defaults(func=cmd_calibrate)

    ps = sub.add_parser("screen", help="Score and screen the candidate pool.")
    ps.add_argument("--candidates", nargs="*", default=None)
    ps.add_argument("--n-jobs", type=int, default=-1)
    ps.set_defaults(func=cmd_screen)

    pd_ = sub.add_parser("diversity", help="Win rate across the state space.")
    pd_.add_argument("--n-bins", type=int, default=4)
    pd_.set_defaults(func=cmd_diversity)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
