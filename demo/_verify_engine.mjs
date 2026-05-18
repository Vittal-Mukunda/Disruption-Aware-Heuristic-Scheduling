// Headless verification of the pure-JS DAHS inference engine against the
// Python reference cases in dahs_model_bundle.json.
//
// Run:  node demo/_verify_engine.mjs
//
// The DAHS object below is the exact engine that gets pasted into
// demo/dahs_dashboard.jsx. Keep the two copies identical.

import { readFileSync } from "node:fs";

const MODEL = JSON.parse(readFileSync(new URL("./dahs_model_bundle.json", import.meta.url)));

// ===========================================================================
// DAHS inference engine -- pure functions, mirror of baselines/ours.py
// ===========================================================================
const DAHS = {
  softmax(v) {
    let mx = -Infinity;
    for (const x of v) if (x > mx) mx = x;
    let s = 0;
    const e = v.map((x) => { const z = Math.exp(x - mx); s += z; return z; });
    return e.map((z) => z / s);
  },

  argmax(v) {
    let bi = 0;
    for (let i = 1; i < v.length; i++) if (v[i] > v[bi]) bi = i;
    return bi;
  },

  // --- regime GMM: GaussianMixture.predict_proba, covariance_type="full" ---
  gmmPosterior(m, state25) {
    const { means, precisionsCholesky, logWeights, logDet } = m.gmm;
    const K = means.length, D = 25;
    const LOG2PI = Math.log(2 * Math.PI);
    const weighted = new Array(K);
    for (let k = 0; k < K; k++) {
      const mu = means[k], pc = precisionsCholesky[k];
      let sumsq = 0;
      for (let j = 0; j < D; j++) {
        let y = 0;
        for (let i = 0; i < D; i++) y += (state25[i] - mu[i]) * pc[i][j];
        sumsq += y * y;
      }
      weighted[k] = -0.5 * (D * LOG2PI + sumsq) + logDet[k] + logWeights[k];
    }
    return DAHS.softmax(weighted);
  },

  // --- XGBoost multi:softprob raw margins (sum of leaf values per class) ---
  // xgboost casts inputs to float32 internally, so we Math.fround the feature
  // vector once -- this makes the tree walk bit-identical to xgboost and shap.
  xgbMargins(m, x) {
    const xf = x.map(Math.fround);
    const margins = m.baseMargin.slice();
    const trees = m.trees, tc = m.treeClass;
    for (let t = 0; t < trees.length; t++) {
      const tr = trees[t], f = tr.f, thr = tr.t, left = tr.l;
      let i = 0;
      while (left[i] !== -1) i = xf[f[i]] < thr[i] ? left[i] : left[i] + 1;
      margins[tc[t]] += thr[i];
    }
    return margins;
  },

  // --- per-class isotonic calibration (out_of_bounds="clip") then renormalize ---
  interpClip(x, xs, ys) {
    const n = xs.length;
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid; else hi = mid;
    }
    const span = xs[hi] - xs[lo];
    if (span <= 0) return ys[lo];
    return ys[lo] + ((x - xs[lo]) / span) * (ys[hi] - ys[lo]);
  },

  calibrate(m, probsXgb) {
    const iso = m.isotonic;
    const cal = probsXgb.map((p, c) => DAHS.interpClip(p, iso[c].x, iso[c].y));
    const s = cal.reduce((a, b) => a + b, 0);
    return s > 0 ? cal.map((v) => v / s) : cal.map(() => 0.25);
  },

  // --- FEFO mask (labeling/soft_label_converter.py fefo_mask) ---
  fefoMask(m, probsCal, pctPerishable) {
    if (pctPerishable >= m.constants.fefoMaskThreshold) return probsCal.slice();
    const out = probsCal.slice();
    out[1] = 0;
    const s = out.reduce((a, b) => a + b, 0);
    if (s < 1e-12) return [1 / 3, 0, 1 / 3, 1 / 3];
    return out.map((v) => v / s);
  },

  entropy(p) {
    let e = 0;
    for (const v of p) if (v > 0) e -= v * Math.log(v);
    return e;
  },

  // --- switching controller (models/switching_controller.py select) ---
  switchingStep(m, masked, ctrl) {
    const tMin = m.constants.tMinIntervals;
    const gate = m.constants.entropyGateRatio;
    const entropyMax = Math.log(masked.length);
    const candidate = DAHS.argmax(masked);
    let chosen;
    if (ctrl.current === null || ctrl.dwell <= 0) {
      chosen = candidate;
    } else {
      chosen = DAHS.entropy(masked) < gate * entropyMax ? candidate : ctrl.current;
    }
    const switched = chosen !== ctrl.current;
    const dwell = switched ? Math.max(tMin - 1, 0) : Math.max(ctrl.dwell - 1, 0);
    return { chosen, switched, controllerNext: { current: chosen, dwell } };
  },

  // --- exact path-dependent TreeSHAP for one tree ---
  // Decomposed by leaf: each root->leaf path has <=4 splits, so only the <=4
  // path features matter. For each leaf we enumerate 2^p (p<=4) feature subsets
  // and apply the Shapley formula. The per-tree phi sum to f(x)-E[f] exactly
  // (Shapley efficiency), and repeated features are handled uniformly. This
  // reproduces shap's tree_path_dependent values.
  _fact: [1, 1, 2, 6, 24, 120],

  shapWeight(a, p) {
    // |A|! (p-|A|-1)! / p!
    return (DAHS._fact[a] * DAHS._fact[p - a - 1]) / DAHS._fact[p];
  },

  shapLeaf(leafVal, path, phi) {
    const uniq = [], pos = {};
    for (const nd of path) {
      if (pos[nd.feat] === undefined) { pos[nd.feat] = uniq.length; uniq.push(nd.feat); }
    }
    const p = uniq.length;
    if (p === 0) return;
    const nSub = 1 << p;
    // W[mask] = reach-probability of this leaf when features in `mask` are known
    const W = new Float64Array(nSub);
    for (let mask = 0; mask < nSub; mask++) {
      let w = 1.0;
      for (const nd of path) {
        if (mask & (1 << pos[nd.feat])) { if (!nd.ok) { w = 0; break; } }
        else w *= nd.cf;
      }
      W[mask] = w;
    }
    for (let fi = 0; fi < p; fi++) {
      const bit = 1 << fi;
      let contrib = 0;
      for (let A = 0; A < nSub; A++) {
        if (A & bit) continue;
        let a = 0;
        for (let tmp = A; tmp; tmp >>= 1) a += tmp & 1;
        contrib += DAHS.shapWeight(a, p) * (W[A | bit] - W[A]);
      }
      phi[uniq[fi]] += leafVal * contrib;
    }
  },

  shapOneTree(tree, x, phi) {
    const { f, t, l, c } = tree;
    const stack = [[0, []]];
    while (stack.length) {
      const [node, path] = stack.pop();
      if (l[node] === -1) { DAHS.shapLeaf(t[node], path, phi); continue; }
      const left = l[node], cNode = c[node];
      const xLeft = x[f[node]] < t[node];
      stack.push([left, path.concat({ feat: f[node], cf: c[left] / cNode, ok: xLeft })]);
      stack.push([left + 1, path.concat({ feat: f[node], cf: c[left + 1] / cNode, ok: !xLeft })]);
    }
  },

  treeShap(m, full31, classIdx) {
    const xf = full31.map(Math.fround);
    const phi = new Array(full31.length).fill(0);
    const trees = m.trees, tc = m.treeClass;
    for (let t = 0; t < trees.length; t++) {
      if (tc[t] === classIdx) DAHS.shapOneTree(trees[t], xf, phi);
    }
    return phi;
  },

  // --- full per-interval inference (stateless part) ---
  infer(m, state25) {
    const regime = DAHS.gmmPosterior(m, state25);
    const full = state25.concat(regime);
    const margins = DAHS.xgbMargins(m, full);
    const probsXgb = DAHS.softmax(margins);
    const probsCal = DAHS.calibrate(m, probsXgb);
    const probsMasked = DAHS.fefoMask(m, probsCal, state25[4]);
    return { regime, full, probsXgb, probsCal, probsMasked };
  },
};

// ===========================================================================
// Verification
// ===========================================================================
let failures = 0;
const maxErr = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

function check(label, ok, detail = "") {
  if (!ok) { failures++; console.log(`  FAIL  ${label}  ${detail}`); }
  else console.log(`  ok    ${label}`);
}

console.log(`\nBundle: ${MODEL.trees.length} trees, ${MODEL.referenceCases.length} reference cases\n`);

console.log("Reference cases (GMM -> XGB -> calibrate -> FEFO mask -> SHAP):");
let probErr = 0, shapErr = 0, shapRel = 0, fullErr = 0, shapErrStored = 0;
for (let i = 0; i < MODEL.referenceCases.length; i++) {
  const rc = MODEL.referenceCases[i];
  const r = DAHS.infer(MODEL, rc.state25);
  probErr = Math.max(probErr, maxErr(r.probsXgb, rc.probsXgb),
    maxErr(r.probsCal, rc.probsCal), maxErr(r.probsMasked, rc.probsMasked));
  fullErr = Math.max(fullErr, maxErr(r.full, rc.full31));
  const chosen = DAHS.argmax(r.probsMasked);
  if (chosen !== rc.chosenClass) {
    failures++;
    console.log(`  FAIL  case ${i} chosenClass ${chosen} != ${rc.chosenClass}`);
  }
  // SHAP on the reconstructed full31
  const shap = DAHS.treeShap(MODEL, r.full, rc.chosenClass);
  const e = maxErr(shap, rc.shapValues);
  shapErr = Math.max(shapErr, e);
  shapRel = Math.max(shapRel, e / (Math.max(...rc.shapValues.map(Math.abs)) || 1));
  // SHAP on the exact stored full31 (isolates GMM reconstruction)
  const shapStored = DAHS.treeShap(MODEL, rc.full31, rc.chosenClass);
  shapErrStored = Math.max(shapErrStored, maxErr(shapStored, rc.shapValues));
}
check("probabilities (XGB/calibrated/masked) match Python", probErr < 1e-5, `maxErr=${probErr.toExponential(2)}`);
check("full31 reconstruction matches Python", fullErr < 1e-6, `maxErr=${fullErr.toExponential(2)}`);
check("chosen class matches Python", failures === 0 || probErr < 1e-5);
check("TreeSHAP matches Python shap", shapErr < 1e-3,
  `maxAbsErr=${shapErr.toExponential(2)} maxRelErr=${(shapRel * 100).toFixed(3)}%`);
check("TreeSHAP on stored full31 matches Python", shapErrStored < 1e-3,
  `maxAbsErr=${shapErrStored.toExponential(2)}`);

if (process.env.DEBUG_SHAP) {
  const sm = (a) => a.reduce((x, y) => x + y, 0);
  for (let idx = 0; idx < MODEL.referenceCases.length; idx++) {
    const rc = MODEL.referenceCases[idx];
    const r = DAHS.infer(MODEL, rc.state25);
    const shap = DAHS.treeShap(MODEL, r.full, rc.chosenClass);
    const diffs = shap.map((v, i) => [i, v - rc.shapValues[i]]).filter((d) => Math.abs(d[1]) > 1e-4);
    if (diffs.length) {
      console.log(`  [case ${idx}] chosen=${rc.chosenClass} sum my=${sm(shap).toFixed(6)} py=${sm(rc.shapValues).toFixed(6)}`);
      console.log(`    diffs: ${diffs.map((d) => `f${d[0]}:my=${shap[d[0]].toFixed(5)},py=${rc.shapValues[d[0]].toFixed(5)}`).join("  ")}`);
    }
  }
}

console.log("\nController trace (stateful switching controller over one shift):");
const trace = MODEL.controllerTrace;
let ctrl = { current: null, dwell: 0 };
let traceOk = 0;
const got = [];
for (let i = 0; i < trace.states.length; i++) {
  const r = DAHS.infer(MODEL, trace.states[i]);
  const step = DAHS.switchingStep(MODEL, r.probsMasked, ctrl);
  ctrl = step.controllerNext;
  const ruleName = MODEL.heuristicNames[step.chosen];
  got.push(ruleName);
  if (ruleName === trace.expectedRules[i]) traceOk++;
}
check(`controller trace ${traceOk}/${trace.states.length} intervals`, traceOk === trace.states.length);
if (traceOk !== trace.states.length) {
  console.log(`    expected: ${trace.expectedRules.join(",")}`);
  console.log(`    got     : ${got.join(",")}`);
}

console.log(`\n${failures === 0 ? "ALL CHECKS PASSED" : failures + " CHECK(S) FAILED"}\n`);
process.exit(failures === 0 ? 0 : 1);
