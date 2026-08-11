"""Feature provenance and redundancy analysis for the state representation.

REVIEWER 1, COMMENT 3.a
-----------------------
    "One fundamental part of the algorithm is the state set. The paper lists 25
     features plus 6 regime-membership features. How were they identified (based
     on the literature?) and selected (was there any correlation analysis?)?"

The honest answer for the submitted version is that the features were designed,
not selected, and that no correlation analysis was performed. That omission was
not cosmetic: two of the 25 were degenerate — `time_to_next_expected_carrier`
was constant within a configuration, and `interval_index_in_shift` and
`intervals_remaining` summed to 32 by construction — which made the feature
matrix exactly singular and silently corrupted the regime GMM's BIC sweep
(`regime/regime_discovery.py`). A correlation analysis would have caught both.

This module supplies what was missing, in two parts.

PROVENANCE — where each feature comes from
------------------------------------------
`simulation.state_extractor.FEATURE_PROVENANCE` records, per feature, its group
and the literature or design rationale it derives from. `export_provenance()`
renders it as the Appendix A table so the manuscript claim and the code cannot
drift apart.

REDUNDANCY — whether the set is over-specified
----------------------------------------------
Four diagnostics on the training feature matrix:

  * near-constant columns (variance below tolerance) — these carry no signal and
    make full-covariance density models singular;
  * Pearson and Spearman correlation, with pairs above `|r| = 0.95` flagged;
  * variance inflation factors, computed by regressing each feature on the rest.
    VIF is the diagnostic that catches *multi*-collinearity — a feature can be an
    exact linear combination of three others while correlating only moderately
    with each, which is precisely the `interval_index`/`intervals_remaining`
    case once lags are present;
  * correlation-distance clustering, which groups features that carry the same
    information and nominates one representative per cluster.

The output is a recommendation, not an automatic edit: the manuscript reports
the analysis and states which features were dropped and why. `--top-k` also
emits the parsimonious subset used by the `top5_features` ablation.

    python -m experiments.feature_analysis --train data/train.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.state_extractor import FEATURE_NAMES, FEATURE_PROVENANCE

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "S1_features"
FIG_DIR = REPO_ROOT / "figures" / "S1_features"

CORR_FLAG: float = 0.95      # |r| above this marks a redundant pair
VIF_FLAG: float = 10.0       # conventional multi-collinearity threshold
VAR_TOL: float = 1e-10       # below this a column is treated as constant


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [f"f_{n}" for n in FEATURE_NAMES if f"f_{n}" in df.columns]
    missing = [n for n in FEATURE_NAMES if f"f_{n}" not in df.columns]
    if missing:
        raise ValueError(f"train parquet is missing feature columns: {missing}")
    return df[cols].to_numpy(np.float64), [c[2:] for c in cols]


def near_constant(X: np.ndarray, names: list[str]) -> list[dict]:
    var = X.var(axis=0)
    return [
        {"feature": names[i], "variance": float(var[i])}
        for i in range(X.shape[1])
        if var[i] <= VAR_TOL
    ]


def variance_inflation(X: np.ndarray, names: list[str]) -> pd.DataFrame:
    """VIF_i = 1 / (1 - R^2_i) from regressing feature i on all the others.

    Computed with `lstsq` rather than a statistics package so the analysis adds
    no dependency. Columns with zero variance are reported as infinite VIF
    rather than dividing by zero — that is the correct reading: a constant is
    perfectly predicted by the intercept alone.
    """
    n, d = X.shape
    Xc = X - X.mean(axis=0)
    scale = Xc.std(axis=0)
    rows: list[dict] = []
    for i in range(d):
        if scale[i] <= np.sqrt(VAR_TOL):
            rows.append({"feature": names[i], "r2": 1.0, "vif": float("inf")})
            continue
        y = Xc[:, i]
        A = np.delete(Xc, i, axis=1)
        A = np.hstack([A, np.ones((n, 1))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_tot = float((y**2).sum())
        r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else 1.0
        r2 = min(max(r2, 0.0), 1.0 - 1e-15)
        rows.append({"feature": names[i], "r2": r2, "vif": float(1.0 / (1.0 - r2))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def redundant_pairs(corr: pd.DataFrame, threshold: float = CORR_FLAG) -> list[dict]:
    out: list[dict] = []
    names = list(corr.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(corr.iloc[i, j])
            if abs(r) >= threshold:
                out.append({"a": names[i], "b": names[j], "r": r})
    return sorted(out, key=lambda d: -abs(d["r"]))


def cluster_representatives(corr: pd.DataFrame, threshold: float = CORR_FLAG) -> dict:
    """Group features by correlation distance; nominate one per group.

    Single-linkage agglomeration on `1 - |r|`, cut at `1 - threshold`. Within a
    cluster the representative is the feature with the lowest mean absolute
    correlation to features *outside* the cluster — i.e. the member that carries
    the group's information with least overlap elsewhere.
    """
    names = list(corr.columns)
    absr = corr.abs().to_numpy()
    parent = list(range(len(names)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if absr[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(len(names)):
        groups.setdefault(find(i), []).append(i)

    out: dict[str, dict] = {}
    for root, members in groups.items():
        if len(members) == 1:
            continue
        outside = [k for k in range(len(names)) if k not in members]
        rep = min(
            members,
            key=lambda m: float(absr[m, outside].mean()) if outside else 0.0,
        )
        out[names[rep]] = {
            "members": [names[m] for m in members],
            "dropped": [names[m] for m in members if m != rep],
        }
    return out


def export_provenance() -> str:
    """Appendix A table: feature, group, source. Markdown."""
    lines = ["| # | Feature | Group | Source / rationale |", "|---:|---|---|---|"]
    for i, name in enumerate(FEATURE_NAMES, start=1):
        group, source = FEATURE_PROVENANCE.get(name, ("—", "—"))
        lines.append(f"| {i} | `{name}` | {group} | {source} |")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=REPO_ROOT / "data" / "train.parquet")
    p.add_argument("--top-k", type=int, default=5,
                   help="Also emit the top-k subset for the parsimony ablation.")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "feature_provenance.md").write_text(
        export_provenance(), encoding="utf-8"
    )
    print(f"[features] wrote provenance table for {len(FEATURE_NAMES)} features")

    if not args.train.exists():
        print(f"[features] {args.train} not found — provenance only. "
              f"Run Stage 2 labelling first for the redundancy analysis.")
        return 0

    df = pd.read_parquet(args.train)
    X, names = feature_matrix(df)
    print(f"[features] matrix {X.shape[0]} rows x {X.shape[1]} features")

    pearson = pd.DataFrame(np.corrcoef(X, rowvar=False), index=names, columns=names)
    spearman = pd.DataFrame(X, columns=names).corr(method="spearman")
    vif = variance_inflation(X, names)

    const = near_constant(X, names)
    pairs = redundant_pairs(pearson)
    clusters = cluster_representatives(pearson)
    high_vif = vif[vif["vif"] >= VIF_FLAG]["feature"].tolist()

    pearson.to_parquet(RESULTS_DIR / "corr_pearson.parquet")
    spearman.to_parquet(RESULTS_DIR / "corr_spearman.parquet")
    vif.to_parquet(RESULTS_DIR / "vif.parquet", index=False)

    report = {
        "n_features": len(names),
        "near_constant": const,
        "redundant_pairs_abs_r_ge_0.95": pairs,
        "vif_ge_10": high_vif,
        "correlation_clusters": clusters,
        "recommended_drop": sorted(
            {c["feature"] for c in const}
            | {d for v in clusters.values() for d in v["dropped"]}
        ),
    }
    (RESULTS_DIR / "feature_analysis.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )

    print(f"\n[features] near-constant columns : {[c['feature'] for c in const] or 'none'}")
    print(f"[features] |r| >= {CORR_FLAG} pairs      : {len(pairs)}")
    for d in pairs[:10]:
        print(f"    {d['a']:<32s} ~ {d['b']:<32s} r={d['r']:+.3f}")
    print(f"[features] VIF >= {VIF_FLAG:.0f}              : {high_vif or 'none'}")
    print(f"[features] recommended drop      : {report['recommended_drop'] or 'none'}")

    if args.top_k:
        shap_path = REPO_ROOT / "results" / "E5" / "shap_global_importance.parquet"
        if shap_path.exists():
            imp = pd.read_parquet(shap_path)
            imp_col = next(
                c for c in imp.columns
                if "import" in c.lower() or "shap" in c.lower()
            )
            top = (
                imp.sort_values(imp_col, ascending=False)["feature"]
                .head(args.top_k)
                .tolist()
            )
        else:
            # Before SHAP exists, fall back to the non-collinear features with
            # the highest marginal variance — a defensible provisional ranking
            # that the ablation replaces once E5 has run.
            merged = vif.merge(
                pd.DataFrame({"feature": names, "var": X.var(axis=0)}), on="feature"
            )
            top = (
                merged[merged["vif"] < VIF_FLAG]
                .sort_values("var", ascending=False)["feature"]
                .head(args.top_k)
                .tolist()
            )
        (RESULTS_DIR / f"top{args.top_k}_features.json").write_text(
            json.dumps({"top_k": args.top_k, "features": top}, indent=2),
            encoding="utf-8",
        )
        print(f"[features] top-{args.top_k} subset for the parsimony ablation: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
