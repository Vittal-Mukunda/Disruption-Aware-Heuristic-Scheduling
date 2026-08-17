"""Deliverable A — real-data simulator validation.

Quantitative distributional comparison of the Olist Brazilian E-Commerce trace
against the warehouse simulator's generative draws. Compares distribution SHAPE
(not magnitude — the two operate on different time scales) via 2-sample KS /
Anderson-Darling tests on mean-normalized samples, plus shape statistics
(coefficient of variation, skewness) and QQ-plots.

Honest scope: Olist is e-commerce order data measured in days; the sim is
warehouse-shift dispatching in minutes. Absolute magnitudes are NOT expected to
match. The defensible claim is shape consistency. With n ~ 1e5, KS p-values are
uninformative (any deviation rejects) -- the KS statistic D and the shape
metrics are the evidence; a small-n subsampled KS p is reported alongside.

Outputs:
    figures/A/olist_validation.{png,pdf}
    results/A/validation_stats.csv
    results/A/validation_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

from simulation.warehouse_env import WarehouseEnv

ROOT = Path(__file__).resolve().parents[1]
OLIST = ROOT / "Olist Dataset"
FIG_DIR = ROOT / "figures" / "A"
RES_DIR = ROOT / "results" / "A"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# Olist product categories (Portuguese) treated as perishable. Conservative:
# food + drink only. Flowers reported separately as a sensitivity note.
PERISHABLE_CATS = {"alimentos", "alimentos_bebidas", "bebidas"}

N_SIM_SHIFTS = 250  # matches the training-set shift count
SUBSAMPLE_AD = 50_000  # cap for Anderson-Darling (performance)
SUBSAMPLE_KS_SMALL = 2_000  # interpretable-p KS at modest n
RNG = np.random.default_rng(42)


def load_sim_draws() -> dict[str, np.ndarray | float]:
    """Collect the simulator's generative draws across N_SIM_SHIFTS shifts."""
    cfg = OmegaConf.load(ROOT / "config.yaml")
    inter, proc, sla, perish = [], [], [], []
    for seed in range(N_SIM_SHIFTS):
        env = WarehouseEnv(seed=seed, cfg=cfg)
        orders = env._all_orders
        arr = np.sort(np.array([o.arrival_time for o in orders]))
        if arr.size > 1:
            inter.append(np.diff(arr))
        proc.append(np.array([o.processing_time for o in orders]))
        sla.append(np.array([o.sla_due - o.arrival_time for o in orders]))
        perish.extend(o.is_perishable for o in orders)
    return {
        "inter_arrival": np.concatenate(inter),
        "processing_time": np.concatenate(proc),
        "sla_window": np.concatenate(sla),
        "perishable_frac": float(np.mean(perish)),
    }


def load_olist_draws() -> dict[str, np.ndarray | float]:
    """Extract the analogous quantities from the Olist trace."""
    # The dataset is not vendored (licence, and ~50 MB), so absence is the
    # normal first-run state rather than an error condition.
    # `fit_input_distributions` already says so usefully; a bare
    # FileNotFoundError at the tail of a 16-hour campaign does not.
    missing = [
        f.name for f in (
            OLIST / "olist_orders_dataset.csv",
            OLIST / "olist_order_items_dataset.csv",
            OLIST / "olist_products_dataset.csv",
        ) if not f.exists()
    ]
    if missing:
        raise SystemExit(
            f"Olist file(s) {missing} not found under '{OLIST}'.\n"
            f"Download 'Brazilian E-Commerce Public Dataset by Olist' from "
            f"Kaggle and unzip it there. Figures 7-8 and the input-distribution "
            f"fits (Reviewer 1, 5.b) depend on it; everything else runs "
            f"without it."
        )
    orders = pd.read_csv(
        OLIST / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_estimated_delivery_date",
        ],
    )

    # Inter-arrival: diffs WITHIN each calendar day (the sim models a single
    # shift as homogeneous Poisson; per-day diffs strip the multi-day growth
    # trend, though intra-day seasonality remains -- noted as a limitation).
    o = orders.dropna(subset=["order_purchase_timestamp"]).copy()
    o["date"] = o["order_purchase_timestamp"].dt.date
    inter_parts = []
    for _, g in o.groupby("date"):
        ts = np.sort(g["order_purchase_timestamp"].values).astype("datetime64[s]")
        if ts.size > 1:
            inter_parts.append(np.diff(ts).astype(np.int64) / 60.0)  # minutes
    inter = np.concatenate(inter_parts)
    inter = inter[inter > 0]

    # Processing time proxy: purchase -> approved latency.
    proc = (
        orders["order_approved_at"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds().to_numpy() / 60.0
    proc = proc[np.isfinite(proc) & (proc > 0)]

    # SLA window proxy: purchase -> estimated delivery date.
    sla = (
        orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds().to_numpy() / 60.0
    sla = sla[np.isfinite(sla) & (sla > 0)]

    # Perishable fraction: an order is perishable if ANY item is a food/drink.
    items = pd.read_csv(OLIST / "olist_order_items_dataset.csv")
    prods = pd.read_csv(OLIST / "olist_products_dataset.csv")
    items = items.merge(
        prods[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    items["perish"] = items["product_category_name"].isin(PERISHABLE_CATS)
    perish_frac = float(items.groupby("order_id")["perish"].any().mean())

    return {
        "inter_arrival": inter,
        "processing_time": proc,
        "sla_window": sla,
        "perishable_frac": perish_frac,
    }


def _normalize(x: np.ndarray) -> np.ndarray:
    """Scale to unit mean -- compares shape independent of time scale."""
    return x / np.mean(x)


def compare(name: str, sim: np.ndarray, real: np.ndarray) -> dict:
    """Distributional-shape comparison on mean-normalized samples."""
    s, r = _normalize(sim), _normalize(real)

    ks = stats.ks_2samp(s, r)

    s_small = RNG.choice(s, min(SUBSAMPLE_KS_SMALL, s.size), replace=False)
    r_small = RNG.choice(r, min(SUBSAMPLE_KS_SMALL, r.size), replace=False)
    ks_small = stats.ks_2samp(s_small, r_small)

    s_ad = RNG.choice(s, min(SUBSAMPLE_AD, s.size), replace=False)
    r_ad = RNG.choice(r, min(SUBSAMPLE_AD, r.size), replace=False)
    ad = stats.anderson_ksamp([s_ad, r_ad])

    return {
        "variable": name,
        "n_sim": int(sim.size),
        "n_real": int(real.size),
        "sim_mean": float(np.mean(sim)),
        "real_mean": float(np.mean(real)),
        "sim_cv": float(np.std(sim) / np.mean(sim)),
        "real_cv": float(np.std(real) / np.mean(real)),
        "sim_skew": float(stats.skew(sim)),
        "real_skew": float(stats.skew(real)),
        "ks_stat": float(ks.statistic),
        "ks_p_fulln": float(ks.pvalue),
        "ks_stat_subsampled": float(ks_small.statistic),
        "ks_p_subsampled": float(ks_small.pvalue),
        "ad_stat": float(ad.statistic),
        "ad_p": float(ad.significance_level),
        "wasserstein_norm": float(stats.wasserstein_distance(s, r)),
    }


def make_figure(sim: dict, real: dict, rows: list[dict]) -> None:
    """QQ-plots (row 1) + normalized density overlays (row 2)."""
    variables = ["inter_arrival", "processing_time", "sla_window"]
    titles = ["Inter-arrival time", "Processing time", "SLA / due-date window"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for col, (var, title) in enumerate(zip(variables, titles)):
        s, r = _normalize(sim[var]), _normalize(real[var])
        q = np.linspace(1, 99, 99)
        sq, rq = np.percentile(s, q), np.percentile(r, q)

        ax = axes[0, col]
        lim = max(sq.max(), rq.max())
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (perfect match)")
        ax.scatter(sq, rq, s=14, alpha=0.7, color="#2c7fb8")
        ax.set_xlabel("Simulator quantile (mean-normalized)")
        ax.set_ylabel("Olist quantile (mean-normalized)")
        ax.set_title(f"QQ — {title}")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        bins = np.linspace(0, np.percentile(np.concatenate([s, r]), 99), 60)
        ax.hist(s, bins=bins, density=True, alpha=0.55, label="Simulator", color="#2c7fb8")
        ax.hist(r, bins=bins, density=True, alpha=0.55, label="Olist", color="#de2d26")
        ax.set_xlabel("Mean-normalized value")
        ax.set_ylabel("Density")
        row = next(x for x in rows if x["variable"] == var)
        ax.set_title(f"{title}\nKS D={row['ks_stat']:.3f}  W={row['wasserstein_norm']:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Deliverable A — Olist trace vs. simulator draws (distribution shape, mean-normalized)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "olist_validation.png", dpi=150)
    fig.savefig(FIG_DIR / "olist_validation.pdf")
    plt.close(fig)


def main() -> None:
    print(f"[A] loading simulator draws ({N_SIM_SHIFTS} shifts)...")
    sim = load_sim_draws()
    print("[A] loading Olist trace...")
    real = load_olist_draws()

    rows = [
        compare(v, sim[v], real[v])
        for v in ["inter_arrival", "processing_time", "sla_window"]
    ]

    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "validation_stats.csv", index=False)

    payload = {
        "continuous": rows,
        "perishable_frac": {
            "sim": sim["perishable_frac"],
            "olist_food_drink": real["perishable_frac"],
            "abs_diff": abs(sim["perishable_frac"] - real["perishable_frac"]),
        },
        "notes": {
            "scale": "Olist measured in days, sim in minutes; tests run on "
            "mean-normalized samples to probe shape only.",
            "ks_pvalue": "Full-n KS p is uninformative at n~1e5; rely on KS D, "
            "the subsampled-n KS p, and the CV/skew shape metrics.",
            "inter_arrival": "Olist inter-arrivals computed within each calendar "
            "day; intra-day seasonality remains a known deviation from the "
            "sim's homogeneous-Poisson assumption.",
            "processing_time": "Olist proxy is purchase->approved latency, not "
            "warehouse pick time -- a proxy, stated as a limitation.",
        },
    }
    with open(RES_DIR / "validation_stats.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    make_figure(sim, real, rows)

    print("\n=== Deliverable A — distributional-shape comparison ===")
    for row in rows:
        print(f"\n{row['variable']}:")
        print(f"  n: sim={row['n_sim']:,}  olist={row['n_real']:,}")
        print(f"  mean: sim={row['sim_mean']:.3f}  olist={row['real_mean']:.3f}  (different scales — expected)")
        print(f"  CV:   sim={row['sim_cv']:.3f}  olist={row['real_cv']:.3f}")
        print(f"  skew: sim={row['sim_skew']:.3f}  olist={row['real_skew']:.3f}")
        print(f"  KS D={row['ks_stat']:.4f} (full-n p={row['ks_p_fulln']:.2e})")
        print(f"  KS D={row['ks_stat_subsampled']:.4f} (n={SUBSAMPLE_KS_SMALL} p={row['ks_p_subsampled']:.4f})")
        print(f"  Anderson-Darling stat={row['ad_stat']:.3f} (p={row['ad_p']:.4f})")
        print(f"  Wasserstein (normalized)={row['wasserstein_norm']:.4f}")

    p = payload["perishable_frac"]
    print(f"\nperishable_frac: sim={p['sim']:.3f}  olist(food+drink)={p['olist_food_drink']:.4f}  abs_diff={p['abs_diff']:.3f}")
    print(f"\n[A] artifacts written:\n  {FIG_DIR / 'olist_validation.png'}\n  {RES_DIR / 'validation_stats.csv'}\n  {RES_DIR / 'validation_stats.json'}")


if __name__ == "__main__":
    main()
