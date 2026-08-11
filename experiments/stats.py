"""Statistical inference helpers (bootstrap CI, paired Wilcoxon, BH-FDR).

WHICH METRIC IS PRIMARY (Reviewer 1, 6.c)
-----------------------------------------
    "The authors say in P12 that 'The primary KPI is the SLA-breach rate'.
     However, that is only part of the composite cost, which also weighs
     tardiness and unprocessed orders. The composite cost should therefore be
     the primary metric of comparison, but when analyzing results, the authors
     focus more on the SLA-breach rate. I assume that all the algorithms were
     optimized for composite cost, but even that was unclear."

Both halves of this are correct and both are now fixed at the source rather than
in the prose.

1. THE OBJECTIVE IS PRIMARY. `cfg.experiments.stats.primary_metric` is
   `composite_cost`, and `metric_order()` returns it first in every table, plot
   and significance sweep. The breach rate is a *component* of the objective and
   is reported as one, alongside the other components — tardiness, spoilage and
   unserved work — so a reader can see the decomposition rather than a single
   component promoted to headline. This also removes the incentive that produced
   the accounting problem Reviewer 2 identified: once the objective leads, a
   method cannot look good by improving one component at the expense of another
   that goes unreported.

2. EVERY METHOD OPTIMISES THE SAME OBJECTIVE, and it is now checkable rather
   than assumed. `simulation.cost` defines it once; the rollout labels integrate
   it (`labeling.rollout_labeler`), the PPO reward is its negated per-interval
   increment (`baselines.ppo_fair`), fitted Q-iteration regresses the same
   increment (`baselines.offline_fqi`), LinUCB's payoff is its negation, and the
   rolling-horizon MPC minimises it directly. The static rules do not optimise
   anything and are evaluated against it. In the submitted code the objective
   was written out twice, in the labeller and in the evaluation harness, which
   is precisely why the reviewer could not tell what was being optimised.

The statistical recipe is otherwise unchanged, per `config.yaml`
`experiments.stats`:

  - Bootstrap 95% CI via percentile method (cfg.experiments.stats.bootstrap_resamples
    = 10000 by default).
  - Paired comparisons across methods on the *same* shift seeds via Wilcoxon
    signed-rank (cfg.experiments.stats.test = "wilcoxon").
  - Multiple-comparison correction via Benjamini-Hochberg with q=0.05.

Inputs are the per-shift KPI parquets from Phase 5 / Phase 6 (e.g.
`results/ours.parquet`, `results/scenario_balanced/ours.parquet`). The shape
across methods is identical (50 rows, canonical KPI schema), and rows are
seed-aligned so paired tests are valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


# Canonical KPI names, defined once so no driver re-invents them.
#
# The submitted schema had `sla_breach_rate` (late / *completed*) and
# `mean_cost`. Both are gone: the first hid every unserved order from its own
# denominator (Reviewer 2, 1) and the second named a per-interval average as if
# it were the objective. Anything still referring to them is reading a schema
# that no longer exists, which is a hard error rather than a silent zero.
PRIMARY_METRIC: str = "composite_cost"
FAILURE_METRIC: str = "service_failure_rate"

#: Default reporting set for drivers that do not take a metric list from config.
DEFAULT_METRICS: list[str] = [
    PRIMARY_METRIC,
    FAILURE_METRIC,
    "sla_breach_rate_arrived",
    "mean_tardiness",
    "spoilage_rate",
    "throughput",
    "unserved",
    "picker_utilization",
]

#: Submitted name -> revised name, for reading pre-revision parquets and for the
#: response letter's mapping between the two result sets.
RENAMED_METRICS: dict[str, str] = {
    "sla_breach_rate": "sla_breach_rate_served",
    "mean_cost": "mean_interval_cost",
}


def require_metrics(df: pd.DataFrame, metrics: list[str]) -> None:
    """Fail loudly when a driver asks for a metric the KPI schema no longer has."""
    missing = [m for m in metrics if m not in df.columns]
    if not missing:
        return
    hints = [
        f"{m} -> {RENAMED_METRICS[m]}" for m in missing if m in RENAMED_METRICS
    ]
    raise KeyError(
        f"KPI columns {missing} are not in this result frame. "
        + (f"Renamed in revision: {hints}. " if hints else "")
        + f"Available: {sorted(df.columns)}"
    )


def metric_order(cfg) -> list[str]:
    """Reporting order: the objective first, then its components and context.

    Every table and figure in the paper consumes this, so the metric hierarchy
    is defined in one place (`cfg.experiments.stats`) instead of being decided
    independently at each call site — which is how the submitted paper ended up
    declaring one metric primary in Section 5 and analysing a different one
    throughout Section 6.
    """
    primary = str(cfg.experiments.stats.primary_metric)
    secondary = [str(m) for m in cfg.experiments.stats.secondary_metrics]
    return [primary] + [m for m in secondary if m != primary]


def is_lower_better(metric: str) -> bool:
    """Direction of improvement, so forest plots and rankings orient correctly."""
    higher_better = {"throughput", "picker_utilization", "arrived"}
    return metric not in higher_better


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    n: int
    n_resamples: int

    def as_row(self) -> dict[str, float]:
        return {
            "point": float(self.point),
            "ci_lo": float(self.lo),
            "ci_hi": float(self.hi),
            "n": int(self.n),
            "n_resamples": int(self.n_resamples),
        }


def bootstrap_mean_ci(
    values: np.ndarray,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 1337,
) -> BootstrapCI:
    """Percentile bootstrap CI for the *mean* of `values`."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"),
                           n=0, n_resamples=n_resamples)
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return BootstrapCI(point=float(values.mean()), lo=lo, hi=hi,
                       n=int(n), n_resamples=int(n_resamples))


def bootstrap_paired_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    n_resamples: int = 10000,
    seed: int = 1337,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of a - b (paired)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: a={a.shape} b={b.shape}")
    if a.size == 0:
        return BootstrapCI(point=float("nan"), lo=float("nan"), hi=float("nan"),
                           n=0, n_resamples=n_resamples)
    diffs = a - b
    return bootstrap_mean_ci(diffs, alpha=alpha, n_resamples=n_resamples, seed=seed)


def paired_wilcoxon(
    a: np.ndarray, b: np.ndarray,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> tuple[float, float]:
    """Return (statistic, p_value) for paired Wilcoxon signed-rank.

    NaN-safe: drops rows where either value is NaN. If all differences are zero,
    returns (0.0, 1.0) — Wilcoxon is undefined on degenerate input.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0 or np.all(a == b):
        return 0.0, 1.0
    res = stats.wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """BH-FDR step-up. Returns (reject_mask, p_adjusted) in the input order."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    m = pvalues.shape[0]
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    ranks = np.arange(1, m + 1)
    p_adj_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)
    p_adj = np.empty(m, dtype=np.float64)
    p_adj[order] = p_adj_sorted
    reject = p_adj <= q
    return reject, p_adj


def compare_methods(
    df_long: pd.DataFrame,
    metric: str,
    method_col: str = "method",
    seed_col: str = "shift_seed",
    baseline: str | None = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    q: float = 0.05,
    bootstrap_seed: int = 1337,
) -> pd.DataFrame:
    """Bootstrap CI per method + paired Wilcoxon vs `baseline` + BH-FDR.

    Parameters
    ----------
    df_long
        Long-form DataFrame with columns at least `[method_col, seed_col, metric]`.
        Rows must be seed-aligned across methods (50 shifts each by default).
    metric
        Column to compare (e.g. "composite_cost").
    baseline
        Method name to use as the reference for paired Wilcoxon. If None,
        only per-method CIs are reported (no pairwise p-values).
    Returns a DataFrame with one row per method.
    """
    methods = list(df_long[method_col].unique())
    if baseline is not None and baseline not in methods:
        raise ValueError(f"baseline '{baseline}' not in methods {methods}")

    pivot = df_long.pivot_table(
        index=seed_col, columns=method_col, values=metric, aggfunc="mean"
    )
    # A paired test needs every method on every seed. Dropping the incomplete
    # rows is right, but doing it SILENTLY is not: if one method was evaluated
    # with --n-test the comparison quietly shrinks to the intersection and still
    # reports as though it covered the corpus.
    n_before = len(pivot)
    pivot = pivot.dropna(axis=0, how="any")
    n_dropped = n_before - len(pivot)
    if n_dropped:
        print(f"[stats] WARNING: {metric}: dropped {n_dropped}/{n_before} seeds "
              f"not present for every method; {len(pivot)} seeds remain paired.")
    if len(pivot) == 0:
        raise ValueError(
            f"No seed is shared by all of {methods} for metric '{metric}'. "
            f"The per-method parquets were produced on different test corpora — "
            f"note that inserting the calibration block shifted the test seed "
            f"range, so pre-revision results/ files do not align with new ones."
        )

    rows: list[dict] = []
    raw_p: list[float] = []
    for m in methods:
        vals = pivot[m].to_numpy(dtype=np.float64)
        ci = bootstrap_mean_ci(vals, alpha=alpha, n_resamples=n_resamples,
                               seed=bootstrap_seed)
        row: dict = {"method": m, "metric": metric, **ci.as_row()}
        if baseline is not None and m != baseline:
            base = pivot[baseline].to_numpy(dtype=np.float64)
            stat, p = paired_wilcoxon(vals, base)
            diff_ci = bootstrap_paired_diff_ci(
                vals, base, alpha=alpha, n_resamples=n_resamples,
                seed=bootstrap_seed,
            )
            row.update({
                "wilcoxon_stat": stat,
                "p_raw": p,
                "diff_point": diff_ci.point,
                "diff_ci_lo": diff_ci.lo,
                "diff_ci_hi": diff_ci.hi,
            })
            raw_p.append(p)
        elif baseline is not None and m == baseline:
            row.update({"wilcoxon_stat": np.nan, "p_raw": np.nan,
                        "diff_point": 0.0, "diff_ci_lo": 0.0, "diff_ci_hi": 0.0})
        rows.append(row)

    df = pd.DataFrame(rows)
    if baseline is not None and raw_p:
        non_baseline_mask = df["method"] != baseline
        reject, p_adj = benjamini_hochberg(np.asarray(raw_p), q=q)
        df.loc[non_baseline_mask, "p_adj_bh"] = p_adj
        df.loc[non_baseline_mask, "reject_bh"] = reject
        df.loc[~non_baseline_mask, "p_adj_bh"] = np.nan
        df.loc[~non_baseline_mask, "reject_bh"] = False
    return df


def load_phase5_results(
    results_dir,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Load per-method parquets into one long-form frame for `compare_methods`.

    Each parquet carries the KPI schema of `experiments.evaluate.KPI_COLUMNS`;
    a `method` column is appended.

    Methods with no parquet on disk are skipped WITH A WARNING. Silently
    skipping them is how a comparison table ends up missing the baseline it was
    meant to beat while still looking complete.
    """
    from pathlib import Path
    results_dir = Path(results_dir)
    if methods is None:
        methods = [p.stem for p in sorted(results_dir.glob("*.parquet"))]
    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for m in methods:
        path = results_dir / f"{m}.parquet"
        if not path.exists():
            missing.append(m)
            continue
        parts.append(pd.read_parquet(path).assign(method=m))
    if missing:
        print(f"[stats] WARNING: no results under {results_dir} for {missing}; "
              f"they are absent from the comparison. Run them, or pass "
              f"--methods explicitly so the omission is deliberate.")
    if not parts:
        raise FileNotFoundError(f"no parquet results found under {results_dir}")
    return pd.concat(parts, axis=0, ignore_index=True)
