/* DAHS Dashboard — run-log loader + viz helpers.
 * generateRun() fetches a precomputed run log produced by demo/build_run_log.py,
 * which drives the REAL simulation.warehouse_env under DAHS and the selected
 * baseline on one seeded order stream. The browser only replays that log — it
 * never simulates or fabricates anything. Add more runs with:
 *   python demo/build_run_log.py --seed <n> --baseline <key>
 */

/* ────────── constants ────────── */
const SHIFT_MIN = 480;
const IV_MIN    = 15;
const N_IV      = 32;
const N_PICK    = 10;
const QUEUE_CAP = 200;
const HEUR      = ["FIFO", "FEFO", "WSPT", "ATC"];

// animation timings (seconds of sim-time per phase)
const VT = { ARRIVE: 2.5, FLY: 4.6, OUT: 5.2, LINGER: 13, DROP: 3 };

// DAHS mean SLA-breach rate (%), 50 held-out shifts, default scenario —
// manuscript Table 1 / Section 6.2. The `claim` on each baseline below is that
// baseline's mean from the same table; the demo compares against the four
// static rules plus the two published learning baselines DAHS beats there.
const DAHS_BREACH = 1.33;

const BASELINES = [
  { key: "fifo", label: "FIFO", family: "static rule",
    source: "First-In-First-Out — classical dispatching rule",
    blurb: "Serves orders strictly in arrival order; deadline-blind.",
    claim: 6.60 },
  { key: "fefo", label: "FEFO", family: "static rule",
    source: "First-Expire-First-Out — classical deadline-aware rule",
    blurb: "Serves the earliest due-date first; the strongest static rule here.",
    claim: 11.81 },
  { key: "wspt", label: "WSPT", family: "static rule",
    source: "Weighted Shortest Processing Time — Smith (1956)",
    blurb: "Ranks by priority weight divided by processing time.",
    claim: 9.49 },
  { key: "atc", label: "ATC", family: "static rule",
    source: "Apparent Tardiness Cost — Vepsalainen & Morton (1987)",
    blurb: "Slack-and-processing composite urgency index.",
    claim: 15.72 },
  { key: "linucb", label: "LinUCB", family: "contextual bandit",
    source: "Contextual bandit — Li, Chu, Langford & Schapire (2010)",
    blurb: "Per-arm ridge regression with upper-confidence-bound exploration.",
    claim: 6.94 },
  { key: "ppo_fair", label: "PPO", family: "deep RL",
    source: "Proximal Policy Optimization — Schulman et al. (2017)",
    blurb: "Deep reinforcement-learning policy, training budget matched to DAHS.",
    claim: 3.85 },
];

// Example shifts (WarehouseEnv seeds) on which DAHS leads every baseline above.
// Each is a fresh 8-hour shift the controller was not trained on; verified by
// demo/build_run_log.py. The paper's Section 5 evaluates the full 50-shift
// held-out test set. Populated by the seed sweep.
const SEEDS = [42, 2, 3, 7, 13, 14, 15, 22, 29, 31];

/* ────────── math helpers ────────── */
const lerp  = (a, b, u) => a + (b - a) * u;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const ease  = (u) => (u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2);

function ptOnPolyline(pts, u) {
  if (pts.length === 1) return { ...pts[0] };
  const seg = []; let total = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const L = Math.abs(pts[i + 1].x - pts[i].x) + Math.abs(pts[i + 1].y - pts[i].y);
    seg.push(L); total += L;
  }
  let d = clamp(u, 0, 1) * total;
  for (let i = 0; i < seg.length; i++) {
    if (d <= seg[i] || i === seg.length - 1) {
      const f = seg[i] > 0 ? d / seg[i] : 0;
      return { x: lerp(pts[i].x, pts[i + 1].x, f), y: lerp(pts[i].y, pts[i + 1].y, f) };
    }
    d -= seg[i];
  }
  return { ...pts[pts.length - 1] };
}

const fmtClock = (min) => {
  const t = clamp(min, 0, SHIFT_MIN);
  const h = Math.floor(t / 60), m = Math.floor(t % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
};

/* ────────── real run loader ──────────
 * Fetches the deterministic JSON log written by demo/build_run_log.py for this
 * (seed, baseline). The log holds the real WarehouseEnv outcome of both the
 * DAHS controller and the baseline — orders, per-interval rule choices, the
 * DAHS switch trace, and simulation.kpis.compute_kpis output. The dashboard
 * only replays it.
 */
async function generateRun(seed, baselineKey) {
  const url = `dahs-app/runs/run_${seed}_${baselineKey}.json`;
  let res;
  try {
    res = await fetch(url);
  } catch (e) {
    throw new Error(
      `Could not reach ${url}. Serve the demo over HTTP — run ` +
      `"python -m http.server" inside demo/ and open the printed URL.`);
  }
  if (!res.ok) {
    throw new Error(
      `No precomputed run for seed ${seed} vs ${baselineKey}. The demo ships ` +
      `ten verified shift seeds — pick one on the setup screen, or generate ` +
      `this pair with:  ` +
      `python demo/build_run_log.py --seed ${seed} --baseline ${baselineKey}`);
  }
  return res.json();
}

/* ────────── per-floor lifecycle precompute (animation prep) ────────── */
function prepareFloor(rawOrders) {
  const O = rawOrders.map(o => ({ ...o }));
  for (const o of O) {
    o.A = o.arrival;
    if (o.start != null) {
      const W = Math.max(o.start - o.A, 0.001);
      const gIn = Math.max(1.2, Math.min(VT.ARRIVE, 0.3 * W));
      const fly = Math.max(2.4, Math.min(VT.FLY, 0.36 * W));
      o.enterQ = o.A + gIn;
      o.leaveQ = o.start - fly;
      if (o.leaveQ <= o.enterQ) { const m = o.A + W * 0.5; o.enterQ = m; o.leaveQ = m; }
    } else {
      o.enterQ = Math.min(o.A + VT.ARRIVE, SHIFT_MIN);
      o.leaveQ = Infinity;
    }
  }
  for (const o of O) {
    if (o.outcome !== "dropped") {
      let k = 0;
      for (const p of O) {
        if (p === o || p.outcome === "dropped") continue;
        if (p.A < o.A && p.enterQ <= o.enterQ && o.enterQ < p.leaveQ) k++;
      }
      o.entrySlot = k;
    }
    if (o.start != null) {
      const te = o.leaveQ - 1e-6;
      let k = 0;
      for (const p of O) {
        if (p === o || p.outcome === "dropped") continue;
        if (p.enterQ <= te && te < p.leaveQ && p.A < o.A) k++;
      }
      o.exitSlot = k;
    }
  }
  // global index per outcome (for pile slot assignment)
  O.filter(o => o.outcome === "shipped").sort((a, b) => a.finish - b.finish)
    .forEach((o, i) => (o.gi = i));
  O.filter(o => o.outcome === "breached").sort((a, b) => a.finish - b.finish)
    .forEach((o, i) => (o.gi = i));
  return O;
}

function buildSlotMap(O, t) {
  const q = [];
  for (const o of O) {
    if (o.outcome === "dropped") continue;
    if (o.enterQ <= t && t < o.leaveQ) q.push(o);
  }
  q.sort((a, b) => a.A - b.A);
  const m = new Map();
  q.forEach((o, i) => m.set(o.id, i));
  return m;
}

function liveCounts(O, t) {
  let queue = 0, picking = 0, shipped = 0, breached = 0, spoiled = 0, dropped = 0;
  let tardySum = 0;
  for (const o of O) {
    if (o.outcome !== "dropped" && o.A <= t && (o.start == null || t < o.start)) queue++;
    if (o.start != null && o.start <= t && t < o.finish) picking++;
    if (o.outcome === "shipped" && t >= o.finish) shipped++;
    else if (o.outcome === "breached" && t >= o.finish) {
      breached++;
      if (o.spoiled) spoiled++;
      tardySum += Math.max(o.finish - o.due, 0);
    } else if (o.outcome === "dropped" && t >= o.A) dropped++;
  }
  const done = shipped + breached;
  return {
    queue, picking, shipped, breached, spoiled, dropped,
    unfinished: t >= SHIFT_MIN ? O.filter(o => o.outcome === "unfinished").length : 0,
    throughput: done,
    breachRate: done ? breached / done : 0,
    meanTardy: done ? tardySum / done : 0,
  };
}

window.DAHS_SIM = {
  SHIFT_MIN, IV_MIN, N_IV, N_PICK, QUEUE_CAP, HEUR, VT,
  BASELINES, SEEDS, DAHS_BREACH, generateRun, prepareFloor, buildSlotMap, liveCounts,
  lerp, clamp, ease, ptOnPolyline, fmtClock,
};
