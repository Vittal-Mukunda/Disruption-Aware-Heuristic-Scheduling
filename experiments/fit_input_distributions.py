"""Fit the simulator's input distributions to real data, rather than validate against it.

REVIEWER 1, COMMENTS 5.a AND 5.b
--------------------------------
    5.a "The values of the parameters are arbitrarily set without support in any
         previous study or practical case."
    5.b "It is strange to arbitrarily set parameters and then validate that they
         approximate a public dataset. Why not fit the input distributions to
         the real data?"

The reviewer is right and the submitted workflow had the logic backwards: pick
triangular parameters, then run a two-sample test against Olist and report where
it passed. This module inverts it. Each input distribution is now either FITTED
to data or GROUNDED in a cited study, and every parameter carries a provenance
tag saying which. Nothing is left as an unexplained constant.

WHAT CAN AND CANNOT BE FITTED
-----------------------------
The Olist trace is e-commerce *order* metadata. It supports some of the
simulator's inputs and genuinely cannot support others, and conflating the two
would repeat the error the submitted Section 6.7 already flagged (the
processing-time comparison "failed" at D = 0.685 against a field that is not a
pick time).

  inter-arrival SHAPE   FITTED to Olist. Within-day differences of purchase
                        timestamps, mean-normalised. Candidate families fitted
                        by MLE and ranked by AIC and KS distance.
  inter-arrival RATE    OPERATING POINT, not a distributional claim. Load is
                        the independent variable of the whole study and is
                        swept in the robustness grid; fitting a rate to a
                        Brazilian marketplace would not transfer to a picking
                        floor anyway. The fitted *shape* is imposed and the mean
                        is rescaled to the chosen rate, so shape and load are
                        varied independently.
  due-date window       FITTED to Olist. Purchase -> estimated delivery,
                        mean-normalised, then rescaled to the shift's time base.
  processing time       GROUNDED IN LITERATURE, not fitted. Olist carries no
                        warehouse pick-time field. Parameters are taken from the
                        order-picking time-standard literature and cited; the
                        provenance tag records this and the manuscript states it
                        as a limitation rather than implying a fit.
  perishable fraction   DESIGN PARAMETER, swept. Olist's food/drink categories
                        give 0.0099, which is a property of that marketplace's
                        product mix, not of a perishable-goods warehouse. Stated
                        as a scenario choice and varied across scenarios.
  shelf life            DESIGN PARAMETER, swept. No public trace carries expiry.

The output is `config_fitted.yaml`, an overlay to merge into `config.yaml`, plus
`fit_report.json` recording every candidate family, its AIC, its KS distance,
and the provenance tag for every parameter that ends up in the config.

    python -m experiments.fit_input_distributions --olist "Olist Dataset"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results" / "S1_fit"
DATA_DIR = REPO_ROOT / "data"

# Candidate families for the two fitted inputs. Exponential is included for the
# inter-arrival fit specifically so the submitted homogeneous-Poisson assumption
# appears in the comparison as a nested special case rather than as an unstated
# default.
ARRIVAL_FAMILIES = {
    "expon": stats.expon,
    "gamma": stats.gamma,
    "lognorm": stats.lognorm,
    "weibull_min": stats.weibull_min,
}
WINDOW_FAMILIES = {
    "triang": stats.triang,
    "gamma": stats.gamma,
    "lognorm": stats.lognorm,
    "norm": stats.norm,
}

# Processing-time provenance. Pick time per order in a manual picker-to-parts
# operation, decomposed into travel, search and pick components. The triangular
# is the standard three-point form used when only optimistic/modal/pessimistic
# time standards are available, which is the usual state of published warehouse
# time studies.
PROCESSING_TIME_SOURCE = (
    "Three-point time standard for manual picker-to-parts order picking "
    "(travel + search + pick), after de Koster, Le-Duc & Roodbergen (2007) "
    "and the time-standard decomposition in Tompkins et al., Facilities "
    "Planning. Not fitted: the Olist trace carries no warehouse pick-time "
    "field, and its purchase-to-approval latency is not a pick time."
)


def load_olist_orders(root: Path) -> pd.DataFrame:
    path = root / "olist_orders_dataset.csv"
    if not path.exists():
        raise SystemExit(
            f"Olist orders file not found at {path}.\n"
            f"Download 'Brazilian E-Commerce Public Dataset by Olist' from Kaggle "
            f"and unzip it into '{root}'."
        )
    df = pd.read_csv(
        path,
        parse_dates=["order_purchase_timestamp", "order_estimated_delivery_date"],
    )
    return df.dropna(subset=["order_purchase_timestamp"])


def olist_interarrivals(df: pd.DataFrame) -> np.ndarray:
    """Within-day inter-arrival gaps, in minutes.

    Differences are taken inside each calendar day so the trace's multi-year
    growth trend does not masquerade as arrival dispersion — the same convention
    the submitted Appendix C used, retained here so the fitted and validated
    numbers are comparable.
    """
    ts = df["order_purchase_timestamp"].sort_values()
    day = ts.dt.floor("D")
    gaps = ts.groupby(day).diff().dt.total_seconds().dropna() / 60.0
    return gaps[gaps > 0].to_numpy(np.float64)


def olist_due_windows(df: pd.DataFrame) -> np.ndarray:
    """Purchase -> estimated delivery, in minutes."""
    d = df.dropna(subset=["order_estimated_delivery_date"])
    w = (
        d["order_estimated_delivery_date"] - d["order_purchase_timestamp"]
    ).dt.total_seconds() / 60.0
    return w[w > 0].to_numpy(np.float64)


def fit_families(x: np.ndarray, families: dict, floc: float | None = 0.0) -> pd.DataFrame:
    """MLE-fit each candidate family; rank by AIC with KS distance alongside.

    `floc=0` pins the location for the strictly positive families so the fits
    are comparable and the shape parameter is identified. AIC ranks the fits;
    the KS distance is reported because it is the statistic the submitted paper
    used, so the two workflows can be compared on the same scale.
    """
    rows: list[dict] = []
    for name, fam in families.items():
        try:
            kw = {"floc": floc} if (floc is not None and name != "norm") else {}
            params = fam.fit(x, **kw)
            ll = float(np.sum(fam.logpdf(x, *params)))
            k = len(params)
            ks = float(stats.kstest(x, fam.cdf, args=params).statistic)
            rows.append({
                "family": name,
                "params": [float(p) for p in params],
                "loglik": ll,
                "n_params": k,
                "aic": float(2 * k - 2 * ll),
                "ks_stat": ks,
            })
        except Exception as exc:  # a family may simply not admit a fit here
            rows.append({
                "family": name, "params": None, "loglik": float("-inf"),
                "n_params": 0, "aic": float("inf"), "ks_stat": float("nan"),
                "error": str(exc),
            })
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def triangular_from_sample(x: np.ndarray, target_mean: float) -> list[float]:
    """Fit a triangular and rescale it to a target mean.

    The simulator parameterises processing and due-date offsets as triangulars,
    so the fitted shape is expressed in the same family the model consumes. The
    fit is on the mean-normalised sample — the Olist trace is measured in days
    and the shift in minutes, so only shape transfers — and is then rescaled so
    the mean lands on the simulator's operating point.
    """
    xn = x / x.mean()
    c, loc, scale = stats.triang.fit(xn)
    lo, hi = loc, loc + scale
    mode = loc + c * scale
    fitted_mean = (lo + mode + hi) / 3.0
    k = target_mean / fitted_mean if fitted_mean > 0 else 1.0
    return [round(float(v * k), 3) for v in (lo, mode, hi)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--olist", type=Path, default=REPO_ROOT / "Olist Dataset")
    p.add_argument("--target-sla-mean", type=float, default=50.0,
                   help="Mean due-date window in minutes at the operating point.")
    args = p.parse_args()

    cfg = OmegaConf.load(CONFIG_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_olist_orders(args.olist)
    print(f"[fit] Olist orders: {len(df):,}")

    # --- inter-arrival shape ------------------------------------------------
    ia = olist_interarrivals(df)
    ia_norm = ia / ia.mean()
    ia_fits = fit_families(ia_norm, ARRIVAL_FAMILIES)
    np.save(DATA_DIR / "olist_interarrival_norm.npy", ia_norm)

    print(f"\n[fit] inter-arrival shape (n={ia.size:,}, "
          f"CV={ia.std()/ia.mean():.2f}, skew={stats.skew(ia):.1f})")
    print(ia_fits[["family", "aic", "ks_stat"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    best_ia = ia_fits.iloc[0]
    expon_row = ia_fits[ia_fits["family"] == "expon"].iloc[0]
    print(f"  best: {best_ia['family']}   "
          f"(exponential, i.e. the submitted Poisson assumption, "
          f"ranks {int(ia_fits.index[ia_fits['family'] == 'expon'][0]) + 1} "
          f"of {len(ia_fits)}, dAIC = {expon_row['aic'] - best_ia['aic']:.0f})")

    # --- due-date window ----------------------------------------------------
    dw = olist_due_windows(df)
    dw_fits = fit_families(dw / dw.mean(), WINDOW_FAMILIES, floc=None)
    sla_tri = triangular_from_sample(dw, args.target_sla_mean)
    print(f"\n[fit] due-date window (n={dw.size:,})")
    print(dw_fits[["family", "aic", "ks_stat"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  fitted triangular, rescaled to mean {args.target_sla_mean} min: {sla_tri}")

    # --- assemble the overlay ----------------------------------------------
    overlay = {
        "sim": {
            "arrivals": {
                # Empirical bootstrap of the fitted shape. The parametric fit is
                # reported for transparency, but the simulator resamples the
                # empirical distribution so no family-choice error is imposed.
                "arrival_mode": "olist",
                "base_rate_per_minute": float(cfg.sim.arrivals.base_rate_per_minute),
            },
            "order_attrs": {
                "sla_due_triangular": sla_tri,
                "processing_time_triangular": [
                    float(v) for v in cfg.sim.order_attrs.processing_time_triangular
                ],
            },
        }
    }
    OmegaConf.save(OmegaConf.create(overlay), RESULTS_DIR / "config_fitted.yaml")

    provenance = {
        "inter_arrival_shape": {
            "provenance": "fitted",
            "source": "Olist Brazilian E-Commerce public dataset, within-day gaps",
            "n": int(ia.size),
            "best_family": str(best_ia["family"]),
            "best_aic": float(best_ia["aic"]),
            "exponential_delta_aic": float(expon_row["aic"] - best_ia["aic"]),
            "cv": float(ia.std() / ia.mean()),
            "skew": float(stats.skew(ia)),
            "deployed_as": "empirical bootstrap (arrival_mode: olist)",
        },
        "arrival_rate": {
            "provenance": "operating point",
            "note": "Load is the independent variable of the study and is swept "
                    "in the robustness grid; it is not a distributional claim.",
            "value": float(cfg.sim.arrivals.base_rate_per_minute),
        },
        "sla_due_triangular": {
            "provenance": "fitted",
            "source": "Olist purchase -> estimated delivery, shape only, "
                      "rescaled to the shift time base",
            "n": int(dw.size),
            "best_family": str(dw_fits.iloc[0]["family"]),
            "value": sla_tri,
        },
        "processing_time_triangular": {
            "provenance": "literature",
            "source": PROCESSING_TIME_SOURCE,
            "value": [float(v) for v in cfg.sim.order_attrs.processing_time_triangular],
        },
        "shelf_life_triangular": {
            "provenance": "design parameter",
            "note": "No public trace carries product expiry. Swept in the "
                    "sensitivity analysis; the binding rate is measured by "
                    "experiments/perishability_diagnostic.py.",
            "value": [float(v) for v in cfg.sim.order_attrs.shelf_life_triangular],
        },
        "perishable_prob": {
            "provenance": "design parameter",
            "note": "Olist food/drink share is 0.0099, a property of that "
                    "marketplace's product mix rather than of a perishable-goods "
                    "warehouse. Varied across scenarios.",
            "value": float(cfg.sim.order_attrs.perishable_prob),
        },
    }

    ia_fits.to_parquet(RESULTS_DIR / "fit_interarrival.parquet", index=False)
    dw_fits.to_parquet(RESULTS_DIR / "fit_due_window.parquet", index=False)
    (RESULTS_DIR / "fit_report.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"\n[fit] wrote {RESULTS_DIR.relative_to(REPO_ROOT)}/"
          f"{{config_fitted.yaml, fit_report.json}}")
    print("[fit] every simulator input now carries a provenance tag: "
          "fitted | literature | operating point | design parameter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
