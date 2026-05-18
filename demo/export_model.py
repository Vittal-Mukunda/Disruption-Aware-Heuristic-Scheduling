"""Export the frozen Phase-4 DAHS model to a compact JSON bundle for the
browser demo (demo/dahs_dashboard.jsx).

Reads runs/phase4/ artifacts strictly read-only -- no retraining, no re-tuning.
The bundle lets a pure-JS engine reproduce the exact DAHS decision pipeline:

    state25 -> GMM regime posteriors -> concat(31) -> XGBoost softprob
            -> per-class isotonic calibration -> FEFO mask -> switching controller

Bundle keys:
    trees           list of {f,t,l,r} parallel arrays (one per XGBoost tree)
    treeClass       class index per tree (from tree_info)
    baseMargin      per-class base margin (measured empirically)
    isotonic        4 per-class {x,y} isotonic breakpoint tables
    gmm             {means, precisionsCholesky, logWeights, logDet}
    featureOrder    31 feature-column names (25 state + 6 regime)
    stateFeatureOrder  25 state-feature names
    constants       tMinIntervals, entropyGateRatio, fefoMaskThreshold
    referenceCases  verification cases: state25 -> expected probs + SHAP
    controllerTrace one shift: states -> expected rule sequence

Run:  python demo/export_model.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # so joblib can unpickle repo classes (models.*)
PHASE4 = REPO / "runs" / "phase4"
OUT = REPO / "demo" / "dahs_model_bundle.json"
HEURISTIC_NAMES = ["FIFO", "FEFO", "WSPT", "ATC"]
FEFO_IDX = 1
N_STATE = 25


# --------------------------------------------------------------------------
# Tree walking (pure Python -- mirrors the JS engine; used to measure baseMargin)
# --------------------------------------------------------------------------
def walk_tree(tree: dict, x: np.ndarray) -> float:
    """Return the leaf value of one tree for input x (XGBoost '<' convention).

    Children are paired: the right child is always left+1 (verified at export).
    Feature values are cast to float32 because XGBoost evaluates splits in
    float32 -- this keeps the bundle consistent with xgboost and shap.
    """
    f, t, left = tree["f"], tree["t"], tree["l"]
    i = 0
    while left[i] != -1:
        i = left[i] if np.float32(x[f[i]]) < t[i] else left[i] + 1
    return t[i]


def predict_margins(trees, tree_class, base_margin, x) -> np.ndarray:
    m = np.array(base_margin, dtype=np.float64)
    for ti, tree in enumerate(trees):
        m[tree_class[ti]] += walk_tree(tree, x)
    return m


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - np.max(v))
    return e / e.sum()


# --------------------------------------------------------------------------
# 1. XGBoost trees from model.json (clean, version-independent)
# --------------------------------------------------------------------------
print("Loading runs/phase4/model.json ...")
with open(PHASE4 / "model.json") as fh:
    raw = json.load(fh)

learner = raw["learner"]
gbm_model = learner["gradient_booster"]["model"]
trees_json = gbm_model["trees"]
tree_class = [int(c) for c in gbm_model["tree_info"]]
lmp = learner["learner_model_param"]
num_class = int(lmp["num_class"])
assert num_class == 4, num_class

trees = []
for tj in trees_json:
    left = [int(x) for x in tj["left_children"]]
    right = [int(x) for x in tj["right_children"]]
    for i in range(len(left)):
        assert left[i] == -1 or right[i] == left[i] + 1, "child-pairing invariant broken"
    # split_conditions doubles as the leaf value for leaf nodes
    cond = [float(np.float32(x)) for x in tj["split_conditions"]]
    feat = [int(x) for x in tj["split_indices"]]
    # sum_hessian = node cover, required by path-dependent TreeSHAP
    cover = [float(np.float32(x)) for x in tj["sum_hessian"]]
    # right child is always left+1, so it is not stored
    trees.append({"f": feat, "t": cond, "l": left, "c": cover})

n_trees = len(trees)
print(f"  {n_trees} trees, num_class={num_class}")

# --------------------------------------------------------------------------
# 2. Measure per-class base margin against the official booster
# --------------------------------------------------------------------------
booster = xgb.Booster()
booster.load_model(str(PHASE4 / "model.json"))

rng = np.random.default_rng(0)
probe = rng.normal(size=(8, 31)).astype(np.float64)
official = booster.predict(xgb.DMatrix(probe), output_margin=True)  # (8,4)
leaf_only = np.array([predict_margins(trees, tree_class, [0, 0, 0, 0], x) for x in probe])
base_diff = official - leaf_only
assert np.allclose(base_diff, base_diff[0], atol=1e-4), f"base margin not constant:\n{base_diff}"
base_margin = base_diff.mean(axis=0)
print(f"  base margin per class: {base_margin}")

# self-check: reconstructed softprob matches the booster
recon = np.array([softmax(predict_margins(trees, tree_class, base_margin, x)) for x in probe])
official_proba = softmax(official[0])  # spot check row 0
assert np.allclose(recon[0], official_proba, atol=1e-5), "softprob reconstruction mismatch"
print("  tree reconstruction verified against booster")

# --------------------------------------------------------------------------
# 3. Isotonic calibration tables (per-class)
# --------------------------------------------------------------------------
print("Loading runs/phase4/calibrator.joblib ...")
calibrator = joblib.load(PHASE4 / "calibrator.joblib")
inner = calibrator._calibrator
calibrated = inner.calibrated_classifiers_[0].calibrators
assert len(calibrated) == num_class

isotonic = []
for c, iso in enumerate(calibrated):
    isotonic.append({
        "x": [float(v) for v in iso.X_thresholds_],
        "y": [float(v) for v in iso.y_thresholds_],
    })
    print(f"  class {c}: {len(isotonic[c]['x'])} isotonic breakpoints")


def interp_clip(x: float, xs: list[float], ys: list[float]) -> float:
    """np.interp with clip-on-both-ends -- matches IsotonicRegression(out_of_bounds='clip')."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0:
        return ys[lo]
    w = (x - xs[lo]) / span
    return ys[lo] + w * (ys[hi] - ys[lo])


def calibrate(probs_xgb: np.ndarray) -> np.ndarray:
    cal = np.array([interp_clip(probs_xgb[c], isotonic[c]["x"], isotonic[c]["y"])
                    for c in range(num_class)], dtype=np.float64)
    s = cal.sum()
    return cal / s if s > 0 else np.full(num_class, 1.0 / num_class)


def fefo_mask(probs: np.ndarray, pct_perishable: float, threshold: float) -> np.ndarray:
    if pct_perishable >= threshold:
        return probs.copy()
    out = probs.copy()
    out[FEFO_IDX] = 0.0
    s = out.sum()
    if s < 1e-12:
        out[:] = 0.0
        out[[0, 2, 3]] = 1.0 / 3.0
        return out
    return out / s

# verify our manual calibration matches the real CalibratedRanker
print("Verifying manual calibration vs CalibratedRanker.predict_proba ...")
chk_probs_xgb = booster.predict(xgb.DMatrix(probe))  # softprob
real_cal = calibrator.predict_proba(probe)
manual_cal = np.array([calibrate(p) for p in chk_probs_xgb])
max_err = float(np.max(np.abs(real_cal - manual_cal)))
print(f"  max calibration error: {max_err:.2e}")
assert max_err < 1e-5, "manual isotonic calibration does not match sklearn"

# --------------------------------------------------------------------------
# 4. Regime GMM (full covariance)
# --------------------------------------------------------------------------
print("Loading runs/phase4/regime.joblib ...")
gmm = joblib.load(PHASE4 / "regime.joblib")
assert gmm.covariance_type == "full"
k_regime = int(gmm.n_components)
prec_chol = gmm.precisions_cholesky_  # (K,25,25)
log_det = np.array([np.sum(np.log(np.diag(prec_chol[k]))) for k in range(k_regime)])

gmm_bundle = {
    "means": gmm.means_.astype(np.float64).tolist(),
    "precisionsCholesky": prec_chol.astype(np.float64).tolist(),
    "logWeights": np.log(gmm.weights_).astype(np.float64).tolist(),
    "logDet": log_det.astype(np.float64).tolist(),
}
print(f"  GMM: {k_regime} components, full covariance, {N_STATE}-dim")


def gmm_posterior(state25: np.ndarray) -> np.ndarray:
    """Reproduce GaussianMixture.predict_proba for covariance_type='full'."""
    weighted = np.empty(k_regime)
    for k in range(k_regime):
        diff = state25 - gmm.means_[k]
        y = diff @ prec_chol[k]
        log_prob = -0.5 * (N_STATE * math.log(2 * math.pi) + np.sum(y * y)) + log_det[k]
        weighted[k] = log_prob + gmm_bundle["logWeights"][k]
    return softmax(weighted)

# verify against sklearn
probe25 = rng.normal(size=(5, N_STATE))
for x in probe25:
    a = gmm_posterior(x)
    b = gmm.predict_proba(x.reshape(1, -1))[0]
    assert np.allclose(a, b, atol=1e-9), f"GMM posterior mismatch: {np.max(np.abs(a - b)):.2e}"
print("  GMM posterior reconstruction verified against sklearn")

# --------------------------------------------------------------------------
# 5. Feature order + config constants
# --------------------------------------------------------------------------
meta = joblib.load(PHASE4 / "ranker_meta.joblib")
feature_order = list(meta["feature_cols"])
assert len(feature_order) == 31
state_feature_order = [c[2:] if c.startswith("f_") else c for c in feature_order[:N_STATE]]

with open(REPO / "config.yaml") as fh:
    cfg = yaml.safe_load(fh)
constants = {
    "tMinIntervals": int(cfg["ranker"]["switching"]["t_min_intervals"]),
    "entropyGateRatio": float(cfg["ranker"]["switching"]["entropy_gate_ratio"]),
    "fefoMaskThreshold": float(cfg["heuristics"]["fefo_mask_threshold"]),
}
print(f"  constants: {constants}")

# --------------------------------------------------------------------------
# 6. Full inference pipeline + reference cases
# --------------------------------------------------------------------------
print("Building SHAP explainer ...")
explainer = shap.TreeExplainer(booster)


def shap_for(full31: np.ndarray) -> np.ndarray:
    """Per-class TreeSHAP values, shape (31, 4)."""
    sv = explainer.shap_values(full31.reshape(1, -1))
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)  # (1,31,4)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[0]  # (31,4)
    raise RuntimeError(f"unexpected SHAP shape {sv.shape}")


def infer(state25: np.ndarray) -> dict:
    regime = gmm_posterior(state25)
    full = np.concatenate([state25, regime])
    margins = predict_margins(trees, tree_class, base_margin, full)
    probs_xgb = softmax(margins)
    probs_cal = calibrate(probs_xgb)
    pct_perish = float(state25[4])
    probs_masked = fefo_mask(probs_cal, pct_perish, constants["fefoMaskThreshold"])
    chosen = int(np.argmax(probs_masked))
    return {
        "regime": regime, "full": full, "probsXgb": probs_xgb,
        "probsCal": probs_cal, "probsMasked": probs_masked, "chosen": chosen,
    }


print("Loading data/test.parquet for reference states ...")
df = pd.read_parquet(REPO / "data" / "test.parquet")
state_cols = feature_order[:N_STATE]
states_all = df[state_cols].to_numpy(dtype=np.float64)

# pick diverse reference rows: spread + low/high perishable for FEFO-mask coverage
perish = states_all[:, 4]
low_p = np.where(perish < constants["fefoMaskThreshold"])[0]
high_p = np.where(perish >= constants["fefoMaskThreshold"])[0]
ref_idx = list(np.linspace(0, len(states_all) - 1, 10, dtype=int))
ref_idx += list(low_p[:3]) + list(high_p[:3])
ref_idx = sorted(set(int(i) for i in ref_idx))

reference_cases = []
shap_f32_gap = 0.0
for i in ref_idx:
    s = states_all[i]
    r = infer(s)
    sv = shap_for(r["full"])  # (31,4)
    # diagnostic: SHAP recomputed on a float32-cast input (xgboost is float32)
    sv32 = shap_for(r["full"].astype(np.float32).astype(np.float64))
    shap_f32_gap = max(shap_f32_gap,
                       float(np.max(np.abs(sv[:, r["chosen"]] - sv32[:, r["chosen"]]))))
    reference_cases.append({
        "state25": [float(v) for v in s],
        "full31": [float(v) for v in r["full"]],
        "probsXgb": [float(v) for v in r["probsXgb"]],
        "probsCal": [float(v) for v in r["probsCal"]],
        "probsMasked": [float(v) for v in r["probsMasked"]],
        "chosenClass": r["chosen"],
        "shapValues": [float(v) for v in sv[:, r["chosen"]]],
    })
print(f"  SHAP float64-vs-float32 input gap: {shap_f32_gap:.2e}")
print(f"  {len(reference_cases)} reference cases built")

# --------------------------------------------------------------------------
# 7. Controller trace -- one full shift through the switching controller
# --------------------------------------------------------------------------
def switching_step(masked, ctrl, t_min, gate_ratio):
    """Mirror models/switching_controller.py select()."""
    entropy_max = math.log(num_class)
    candidate = int(np.argmax(masked))
    if ctrl["current"] is None or ctrl["dwell"] <= 0:
        chosen = candidate
    else:
        ent = -sum(p * math.log(p) for p in masked if p > 0)
        chosen = candidate if ent < gate_ratio * entropy_max else ctrl["current"]
    if chosen != ctrl["current"]:
        ctrl["current"] = chosen
        ctrl["dwell"] = max(t_min - 1, 0)
    else:
        ctrl["dwell"] = max(ctrl["dwell"] - 1, 0)
    return chosen


shift_ids = df["shift_id"].unique()
shift0 = df[df["shift_id"] == shift_ids[0]].sort_values("interval_idx")
trace_states = shift0[state_cols].to_numpy(dtype=np.float64)
ctrl = {"current": None, "dwell": 0}
trace_rules = []
for s in trace_states:
    r = infer(s)
    idx = switching_step(r["probsMasked"], ctrl,
                         constants["tMinIntervals"], constants["entropyGateRatio"])
    trace_rules.append(HEURISTIC_NAMES[idx])

# cross-check against the real repo pipeline (baselines/ours.py)
try:
    from baselines.ours import load_ours
    policy = load_ours()
    policy.reset()
    real_rules = [policy(s) for s in trace_states]
    agree = sum(a == b for a, b in zip(trace_rules, real_rules))
    print(f"  controller trace: {agree}/{len(trace_rules)} intervals agree with baselines/ours.py")
    if agree < len(trace_rules):
        print(f"    ours.py: {real_rules}")
        print(f"    bundle : {trace_rules}")
except Exception as exc:  # noqa: BLE001
    print(f"  (could not cross-check against baselines/ours.py: {exc})")

controller_trace = {
    "states": [[float(v) for v in s] for s in trace_states],
    "expectedRules": trace_rules,
}

# --------------------------------------------------------------------------
# 8. Write the bundle
# --------------------------------------------------------------------------
bundle = {
    "trees": trees,
    "treeClass": tree_class,
    "baseMargin": [float(v) for v in base_margin],
    "isotonic": isotonic,
    "gmm": gmm_bundle,
    "featureOrder": feature_order,
    "stateFeatureOrder": state_feature_order,
    "heuristicNames": HEURISTIC_NAMES,
    "constants": constants,
    "referenceCases": reference_cases,
    "controllerTrace": controller_trace,
}

OUT.write_text(json.dumps(bundle, separators=(",", ":")))
size_mb = OUT.stat().st_size / 1e6
print(f"\nWrote {OUT}  ({size_mb:.2f} MB)")
print("Done.")
