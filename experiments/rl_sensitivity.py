"""RL baseline sensitivity and offline action coverage.

REVIEWER 1, COMMENT 6.b
-----------------------
    "Sections 6.8 and 6.9 analyze why the performance of both RL algorithms
     (PPO and offline fitted Q-iteration) is well behind DAHS. The proposed
     explanations are plausible but currently post hoc. Before characterizing
     the results as expected RL failure modes, the authors should report
     sensitivity to the discount factor, GAE parameter, rollout horizon, entropy
     coefficient, and reward normalization; and quantify offline action coverage
     in breach-prone states."

The submitted paper varied exactly one thing about PPO — the training budget,
8k against 500k steps — and concluded from that alone that "the issue is
structural, not budgetary". No hyperparameter was ever tuned, and crucially
`baselines/ppo_fair.py` used no observation or reward normalisation at all, on a
feature vector whose columns span queue lengths in the hundreds and utilisations
in [0, 1]. Unnormalised observations are the single most common cause of a
policy-gradient method failing to learn anything, so the submitted evidence
cannot distinguish "policy gradients are the wrong instrument here" from "this
PPO run was misconfigured". This module removes that ambiguity.

    python -m experiments.rl_sensitivity ppo
    python -m experiments.rl_sensitivity coverage

1. PPO SENSITIVITY
------------------
A one-factor-at-a-time sweep around a stated baseline over exactly the knobs the
reviewer names — discount factor, GAE lambda, rollout horizon (`n_steps`),
entropy coefficient, reward normalisation — plus the full 2x2 on observation and
reward normalisation, because that is the pair most likely to be the actual
cause and a marginal sweep would hide an interaction between them.

The verdict is reported as a number, not a narrative: `gap_closed_fraction` is
how much of the submitted PPO-to-DAHS gap the best configuration recovers. If it
is large, Section 6.9's claim must be withdrawn and rewritten. If it is small
across the whole sweep, the claim is earned and can be stated with the sweep as
evidence.

2. OFFLINE ACTION COVERAGE
--------------------------
Reports state-action visitation restricted to breach-prone states, which is what
determines whether fitted Q-iteration's value estimates there are supported by
data or extrapolated.

This diagnostic exists because the submitted setup had a defect the paper
asserted it did not have. The behaviour policy was `rule = pool[interval_idx %
|H|]`, and `interval_index_in_shift` is an observed feature — so the logged
action was a deterministic function of the state. Marginal coverage was uniform;
CONDITIONAL coverage was degenerate. Section 6.10's claim that round-robin
coverage means "the distribution-shift pathologies that motivate conservative
offline RL do not arise" was therefore unsupported. The corpus now uses a seeded
random behaviour policy (`labeling.rollout_labeler.behaviour_policy`) and this
module quantifies coverage under both, so the paper can report the difference
rather than assume it away.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

from seed import shift_corpora
from simulation.heuristics import resolve_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "E11_rl_sensitivity"

# Stated baseline for the OFAT sweep. These are the submitted `ppo_fair`
# settings, so every row of the sweep is interpretable as a delta from the
# configuration the paper actually reported.
PPO_BASELINE: dict = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "n_steps": 64,
    "ent_coef": 0.0,
    "learning_rate": 3e-4,
    "n_epochs": 4,
    "batch_size": 64,
    "normalize_obs": False,
    "normalize_reward": False,
}


def _sweep_configs(cfg: DictConfig) -> list[dict]:
    """OFAT around `PPO_BASELINE`, plus the full normalisation 2x2."""
    grid = cfg.baselines.ppo_sensitivity
    configs: list[dict] = [dict(PPO_BASELINE, _tag="baseline", _factor="-")]

    for factor in ("gamma", "gae_lambda", "n_steps", "ent_coef"):
        for v in grid[factor]:
            if v == PPO_BASELINE[factor]:
                continue
            c = dict(PPO_BASELINE)
            c[factor] = type(PPO_BASELINE[factor])(v)
            c["_tag"] = f"{factor}={v}"
            c["_factor"] = factor
            configs.append(c)

    # Normalisation is swept jointly: observation scaling changes the effective
    # gradient magnitude and reward scaling changes the advantage scale, and
    # their interaction is exactly what a marginal sweep would miss.
    for n_obs, n_rew in product(grid["normalize_obs"], grid["normalize_reward"]):
        if (n_obs, n_rew) == (False, False):
            continue
        c = dict(PPO_BASELINE)
        c["normalize_obs"], c["normalize_reward"] = bool(n_obs), bool(n_rew)
        c["_tag"] = f"norm(obs={n_obs},rew={n_rew})"
        c["_factor"] = "normalization"
        configs.append(c)
    return configs


def cmd_ppo(args: argparse.Namespace) -> int:
    from baselines.ppo_fair import train_ppo_fair, load_ppo_fair
    from experiments.evaluate import canonical_test_seeds, evaluate_policy

    cfg = OmegaConf.load(CONFIG_PATH)
    seeds = canonical_test_seeds(cfg)[: args.n_test] if args.n_test else canonical_test_seeds(cfg)
    configs = _sweep_configs(cfg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ppo-sensitivity] {len(configs)} configurations x "
          f"{int(cfg.baselines.ppo_fair.total_timesteps):,} steps, "
          f"evaluated on {len(seeds)} test shifts")

    rows: list[dict] = []
    for i, c in enumerate(configs, 1):
        tag = c.pop("_tag")
        factor = c.pop("_factor")
        run_dir = REPO_ROOT / "runs" / "ppo_sensitivity" / tag.replace("/", "_")
        print(f"  [{i:>2d}/{len(configs)}] {tag}")
        try:
            train_ppo_fair(run_dir=run_dir, cfg=cfg, hyperparams=c)
            policy = load_ppo_fair(run_dir)
            df = evaluate_policy(
                f"ppo_{tag}", policy, seeds, cfg,
                results_dir=RESULTS_DIR / "per_config", save=True,
            )
            rows.append({
                "tag": tag, "factor": factor, **c,
                "composite_cost": float(df["composite_cost"].mean()),
                "service_failure_rate": float(df["service_failure_rate"].mean()),
                "throughput": float(df["throughput"].mean()),
                "failed": False,
            })
        except Exception as exc:
            print(f"      FAILED: {exc}")
            rows.append({"tag": tag, "factor": factor, **c, "failed": True,
                         "error": str(exc)})

    table = pd.DataFrame(rows)
    table.to_parquet(RESULTS_DIR / "ppo_sensitivity.parquet", index=False)

    ok = table[~table["failed"]]
    summary: dict = {"n_configs": len(configs), "n_failed": int(table["failed"].sum())}
    if len(ok):
        base = ok[ok["tag"] == "baseline"]
        best = ok.loc[ok["composite_cost"].idxmin()]
        summary["baseline_cost"] = (
            float(base["composite_cost"].iloc[0]) if len(base) else None
        )
        summary["best_tag"] = str(best["tag"])
        summary["best_cost"] = float(best["composite_cost"])
        summary["per_factor_spread"] = {
            f: float(g["composite_cost"].max() - g["composite_cost"].min())
            for f, g in ok.groupby("factor")
        }

        ours = REPO_ROOT / "results" / "ours.parquet"
        if ours.exists() and summary["baseline_cost"] is not None:
            dahs = float(pd.read_parquet(ours)["composite_cost"].mean())
            gap = summary["baseline_cost"] - dahs
            closed = (summary["baseline_cost"] - summary["best_cost"]) / gap if gap else 0.0
            summary["dahs_cost"] = dahs
            summary["gap_closed_fraction"] = float(closed)
            summary["verdict"] = (
                "Tuning closes a substantial share of the gap. Section 6.9's "
                "'structural, not budgetary' claim must be withdrawn and the "
                "tuned configuration reported as the PPO baseline."
                if closed > 0.5 else
                "Tuning does not close the gap across the swept space. The "
                "structural claim is supported and this sweep is the evidence."
            )

    (RESULTS_DIR / "ppo_sensitivity.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + ok[["tag", "factor", "composite_cost", "service_failure_rate"]]
          .sort_values("composite_cost")
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "verdict" in summary:
        print(f"\n[ppo-sensitivity] gap closed: {summary['gap_closed_fraction']:.1%}")
        print(f"[ppo-sensitivity] {summary['verdict']}")
    return 0


# ---------------------------------------------------------------------------
# Offline action coverage
# ---------------------------------------------------------------------------


def _breach_prone_mask(df: pd.DataFrame, pool: list[str], q: float) -> np.ndarray:
    """States from which failure is hard to avoid.

    Defined from the labels rather than from a hand-picked feature rule: a state
    is breach-prone when the BEST achievable rollout cost — the min over rules —
    sits in the top `q` quantile. That is the region where the choice of rule is
    both consequential and difficult, and therefore the region where a value
    function needs data most.
    """
    best = df[[f"cost_{h}" for h in pool]].to_numpy(np.float64).min(axis=1)
    return best >= np.quantile(best, q)


def cmd_coverage(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(CONFIG_PATH)
    pool = resolve_pool(cfg)
    train_path = Path(args.train)
    if not train_path.exists():
        raise SystemExit(f"{train_path} not found; run Stage 2 labelling first.")

    df = pd.read_parquet(train_path)
    if "behaviour_rule" not in df.columns:
        raise SystemExit(
            "train parquet has no `behaviour_rule` column. Regenerate with the "
            "current labeller — the submitted corpus did not record which action "
            "the behaviour policy actually took, which is why the coverage claim "
            "in Section 6.10 was never checked."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mask = _breach_prone_mask(df, pool, args.quantile)
    n_actions = len(pool)

    def _effective_actions(actions: pd.Series) -> float:
        """Exponentiated entropy of the action distribution.

        Equals |H| under a uniform policy and 1.0 when a single action is taken.
        Reported instead of a raw count because it is on the scale of "how many
        actions are actually represented here", which is the quantity a value
        estimate at that state depends on.
        """
        p = (
            actions.value_counts(normalize=True)
            .reindex(pool)
            .fillna(0.0)
            .to_numpy(np.float64)
        )
        nz = p[p > 0]
        return float(np.exp(-(nz * np.log(nz)).sum())) if nz.size else 0.0

    def _counts(sub: pd.DataFrame) -> dict:
        c = sub["behaviour_rule"].value_counts().reindex(pool).fillna(0).astype(int)
        shares = (c / max(int(c.sum()), 1)).to_numpy(np.float64)
        return {
            "n": int(c.sum()),
            "per_action": {h: int(c[h]) for h in pool},
            "min_action_share": float(shares.min()),
            "effective_actions": _effective_actions(sub["behaviour_rule"]),
        }

    overall = _counts(df)
    prone = _counts(df[mask])

    # The conditional test. If the behaviour action is a function of an observed
    # feature, coverage collapses once you condition on that feature even though
    # the marginal looks uniform. `interval_idx` is the feature at issue.
    by_interval = (
        df.groupby("interval_idx")["behaviour_rule"]
        .apply(_effective_actions)
        .rename("effective_actions")
        .reset_index()
    )
    cond_mean = float(by_interval["effective_actions"].mean())

    report = {
        "behaviour_policy": str(cfg.labeling.observed_policy),
        "n_actions": n_actions,
        "breach_prone_quantile": float(args.quantile),
        "overall": overall,
        "breach_prone": prone,
        "conditional_on_interval_idx": {
            "mean_effective_actions": cond_mean,
            "interpretation": (
                "Marginal coverage can be uniform while conditional coverage is "
                "degenerate. Under the submitted round-robin scheme this value "
                "is 1.0 by construction — the logged action is a deterministic "
                "function of interval_idx, which is an observed feature — so "
                "every Q(s, a) for a not equal to the logged action is an "
                "extrapolation, contradicting Section 6.10's coverage claim."
            ),
        },
        "verdict": (
            "Coverage is adequate: the offline-RL deficit cannot be attributed "
            "to unsupported actions in the hard region."
            if (prone["min_action_share"] > 0.5 / n_actions and cond_mean > 0.5 * n_actions)
            else "Coverage is inadequate in the breach-prone region and/or "
            "conditionally degenerate. The offline-RL comparison must be re-run "
            "on a corpus with a non-degenerate behaviour policy before any "
            "conclusion is drawn about value bootstrapping."
        ),
    }
    by_interval.to_parquet(RESULTS_DIR / "coverage_by_interval.parquet", index=False)
    (RESULTS_DIR / "action_coverage.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"[coverage] behaviour policy: {report['behaviour_policy']}  "
          f"|H| = {n_actions}")
    print(f"[coverage] overall      : effective actions "
          f"{overall['effective_actions']:.2f}, min share "
          f"{overall['min_action_share']:.3f}  (n={overall['n']:,})")
    print(f"[coverage] breach-prone : effective actions "
          f"{prone['effective_actions']:.2f}, min share "
          f"{prone['min_action_share']:.3f}  (n={prone['n']:,})")
    print(f"[coverage] conditional on interval_idx: "
          f"{cond_mean:.2f} of {n_actions} effective actions")
    print(f"\n[coverage] {report['verdict']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("ppo", help="OFAT sensitivity sweep for the PPO baseline.")
    pp.add_argument("--n-test", type=int, default=None)
    pp.set_defaults(func=cmd_ppo)

    pc = sub.add_parser("coverage", help="Offline action coverage diagnostics.")
    pc.add_argument("--train", type=Path, default=REPO_ROOT / "data" / "train.parquet")
    pc.add_argument("--quantile", type=float, default=0.75,
                    help="Top quantile of best-achievable cost defining "
                         "'breach-prone'.")
    pc.set_defaults(func=cmd_coverage)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
