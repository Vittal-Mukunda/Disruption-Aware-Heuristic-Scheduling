/* ===========================================================================
 * DAHS — Shop-Floor Simulation  (demo/dahs_dashboard.jsx)
 * ---------------------------------------------------------------------------
 * An interactive academic exhibit: a side-by-side 2D replay of one 8-hour
 * warehouse dispatching shift, DAHS vs. a selectable baseline, on an identical
 * order stream.
 *
 * 100% real / 100% faithful. This component never simulates anything. Pressing
 * "Run" calls the dev-server endpoint /api/run, which executes the REAL Python
 * harness demo/build_run_log.py: the actual simulation.warehouse_env.WarehouseEnv
 * driven by the actual baselines.* policies (DAHS = baselines.ours). The harness
 * asserts its KPIs against simulation.kpis.compute_kpis. The browser only
 * replays the resulting event log, so the animation cannot drift from the
 * paper's simulator. A fixed seed reproduces the run byte-for-byte.
 * ======================================================================== */

import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";

/* ---------------------------------------------------------------------------
 * 0. Fixed simulation structure (config.yaml — single source of truth)
 * ------------------------------------------------------------------------ */
const SHIFT_MIN = 480;          // 8-hour shift
const IV_MIN = 15;              // decision interval
const N_IV = 32;                // intervals per shift
const N_PICK = 10;              // pickers
const QUEUE_CAP = 200;          // queue capacity
const HEUR = ["FIFO", "FEFO", "WSPT", "ATC"];
const VT = { ARRIVE: 2.5, FLY: 4.6, OUT: 5.2, LINGER: 13, DROP: 3 };

// Selectable baselines — mirrors demo/build_run_log.py BASELINES. The "source"
// is the paper / origin each policy comes from.
const BASELINES = [
  { key: "fifo", label: "FIFO", family: "static rule",
    source: "First-In-First-Out — classical dispatching rule",
    blurb: "Serves orders strictly in arrival order; deadline-blind." },
  { key: "fefo", label: "FEFO", family: "static rule",
    source: "First-Expire-First-Out — classical deadline-aware rule",
    blurb: "Serves the earliest due-date first; the strongest static rule here." },
  { key: "wspt", label: "WSPT", family: "static rule",
    source: "Weighted Shortest Processing Time — Smith (1956)",
    blurb: "Ranks by priority weight divided by processing time." },
  { key: "atc", label: "ATC", family: "static rule",
    source: "Apparent Tardiness Cost — Vepsalainen & Morton (1987)",
    blurb: "Slack-and-processing composite urgency index." },
  { key: "greedy_mpc", label: "Greedy MPC", family: "analytic controller",
    source: "One-step-lookahead oracle proxy — this work (cf. Bertsekas, 2020)",
    blurb: "Each interval, simulates every rule one step ahead and picks the cheapest." },
  { key: "snapshot_xgb", label: "Snapshot-XGB", family: "learned ablation",
    source: "DAHS rollout-horizon ablation (tau = 1) — this work",
    blurb: "The DAHS pipeline with its rollout horizon collapsed to a single step." },
  { key: "linucb", label: "LinUCB", family: "contextual bandit",
    source: "Contextual bandit — Li, Chu, Langford & Schapire (2010)",
    blurb: "Per-arm ridge regression with upper-confidence-bound exploration." },
  { key: "ppo_fair", label: "PPO", family: "deep RL",
    source: "Proximal Policy Optimization — Schulman et al. (2017)",
    blurb: "Deep reinforcement-learning policy, training budget matched to DAHS." },
];

/* ---------------------------------------------------------------------------
 * 1. Math helpers
 * ------------------------------------------------------------------------ */
const lerp = (a, b, u) => a + (b - a) * u;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2);

function ptOnPolyline(pts, u) {
  if (pts.length === 1) return { ...pts[0] };
  const seg = [];
  let total = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const L = Math.abs(pts[i + 1].x - pts[i].x) + Math.abs(pts[i + 1].y - pts[i].y);
    seg.push(L);
    total += L;
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
  const h = Math.floor(t / 60);
  const m = Math.floor(t % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
};

/* ---------------------------------------------------------------------------
 * 2. Floor-local geometry (one floor; both floors share it, offset by originY)
 * ------------------------------------------------------------------------ */
const G = {
  W: 1380, H: 516,
  spawn: { x: 46, y: 250 },
  q: { x0: 206, y0: 70, cols: 8, pitch: 17 },
  node: { x: 492, y: 122 },
  fanBusX: 540,
  pickY0: 62, pickPitch: 43, pickInX: 566, pickBoxX: 712, pickGlyphX: 600,
  pickCellX0: 560, pickCellX1: 1018,
  ship: { x0: 1066, y0: 78, cols: 5, pitch: 22 },
  shipDock: { x: 1150, y: 250 },
  breach: { x0: 1060, y0: 316, cols: 18, pitch: 13 },
  dropBin: { x: 318, y: 470 },
};
const pickY = (i) => G.pickY0 + i * G.pickPitch + 21;
const qSlot = (k) => ({
  x: G.q.x0 + (k % G.q.cols) * G.q.pitch + G.q.pitch / 2,
  y: G.q.y0 + Math.floor(k / G.q.cols) * G.q.pitch + G.q.pitch / 2,
});
const gridPos = (g, i) => ({
  x: g.x0 + (i % g.cols) * g.pitch + g.pitch / 2,
  y: g.y0 + Math.floor(i / g.cols) * g.pitch + g.pitch / 2,
});

const STATIONS = [
  { n: 1, key: "arrival", title: "Order arrival", x: 14, w: 176 },
  { n: 2, key: "queue", title: "Order queue", x: 198, w: 224 },
  { n: 3, key: "dispatch", title: "Dispatcher", x: 430, w: 124 },
  { n: 4, key: "pickers", title: "Picker pool", x: 562, w: 474 },
  { n: 5, key: "ontime", title: "Shipped on-time", x: 1044, w: 322, hSplit: true },
];
const PAL = {
  arrival: "#3b6fc9", queue: "#7c5bd0", dispatch: "#1f9d57",
  pickers: "#e07b35", ontime: "#1f9d57", breach: "#d6483c",
};

/* ---------------------------------------------------------------------------
 * 3. Per-floor lifecycle precompute
 * ------------------------------------------------------------------------ */
function prepareFloor(rawOrders) {
  const O = rawOrders.map((o) => ({ ...o }));
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
  O.filter((o) => o.outcome === "shipped").sort((a, b) => a.finish - b.finish)
    .forEach((o, i) => (o.gi = i));
  O.filter((o) => o.outcome === "breached").sort((a, b) => a.finish - b.finish)
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

// floor-local position of one order at sim-time t (all paths are orthogonal)
function orderViz(o, t, slotMap) {
  if (t < o.A) return null;

  if (o.outcome === "dropped") {
    const t1 = o.A + VT.ARRIVE;
    const t2 = t1 + VT.DROP;
    if (t < t1) {
      const u = ease((t - o.A) / VT.ARRIVE);
      const p = ptOnPolyline([G.spawn, { x: G.dropBin.x, y: G.spawn.y }], u);
      return { ...p, s: 1, rot: 0, phase: "arriving", tone: "kraft" };
    }
    if (t < t2) {
      const u = ease((t - t1) / VT.DROP);
      const p = ptOnPolyline([{ x: G.dropBin.x, y: G.spawn.y }, G.dropBin], u);
      return { ...p, s: 1 - 0.4 * u, rot: 0, phase: "drop", tone: "drop" };
    }
    return null;
  }

  if (t < o.enterQ) {
    const u = ease((t - o.A) / Math.max(o.enterQ - o.A, 1e-6));
    const slot = qSlot(o.entrySlot);
    const p = ptOnPolyline([G.spawn, { x: slot.x, y: G.spawn.y }, slot], u);
    return { ...p, s: lerp(0.8, 1, u), rot: 0, phase: "arriving", tone: "kraft" };
  }

  if (t < o.leaveQ) {
    const k = slotMap.get(o.id) ?? o.entrySlot ?? 0;
    const p = qSlot(k);
    const unfinished = o.start == null && t >= SHIFT_MIN;
    return {
      x: p.x, y: p.y + Math.sin(t * 1.1 + o.id * 0.7) * 0.8,
      s: 1, rot: 0, phase: "queued", tone: unfinished ? "unfinished" : "kraft",
    };
  }

  if (o.start != null && t < o.start) {
    const u = ease((t - o.leaveQ) / Math.max(o.start - o.leaveQ, 1e-6));
    const slot = qSlot(o.exitSlot);
    const py = pickY(o.picker);
    // orthogonal: slot -> up to node level -> node -> down to picker level -> picker
    const path = [
      slot,
      { x: slot.x, y: G.node.y },
      { x: G.node.x, y: G.node.y },
      { x: G.node.x, y: py },
      { x: G.pickBoxX, y: py },
    ];
    const p = ptOnPolyline(path, u);
    return { ...p, s: lerp(1, 0.9, u), rot: 0, phase: "dispatch", tone: "active" };
  }

  if (o.start != null && t < o.finish) {
    return {
      x: G.pickBoxX, y: pickY(o.picker), s: 0.9, rot: 0, phase: "processing",
      prog: clamp((t - o.start) / Math.max(o.finish - o.start, 1e-6), 0, 1),
      tone: "active",
    };
  }

  if (o.start != null) {
    const tOut = o.finish + VT.OUT;
    const breached = o.outcome === "breached";
    const py = pickY(o.picker);
    const dest = breached ? gridPos(G.breach, o.gi) : gridPos(G.ship, o.gi % 25);
    if (t < tOut) {
      const u = ease((t - o.finish) / VT.OUT);
      const p = ptOnPolyline(
        [{ x: G.pickBoxX, y: py }, { x: dest.x, y: py }, dest], u);
      return { ...p, s: lerp(0.9, breached ? 0.6 : 0.8, u), rot: 0,
        phase: "outbound", tone: o.spoiled ? "spoiled" : breached ? "breach" : "ship" };
    }
    if (breached) {
      return { x: dest.x, y: dest.y, s: 0.6, rot: 0, phase: "breachPile",
        tone: o.spoiled ? "spoiled" : "breach" };
    }
    if (t < tOut + VT.LINGER) {
      return { x: dest.x, y: dest.y, s: 0.8, rot: 0, phase: "docked", tone: "ship" };
    }
    return null;
  }
  return null;
}

/* ---------------------------------------------------------------------------
 * 4. Cartoon glyphs
 * ------------------------------------------------------------------------ */
const TONE = {
  kraft: { body: "#d8a868", edge: "#9a6a34" },
  active: { body: "#f0c277", edge: "#a87636" },
  ship: { body: "#79cd97", edge: "#2f8f57" },
  breach: { body: "#ec8a6b", edge: "#a8412a" },
  spoiled: { body: "#9f9b66", edge: "#585333" },
  unfinished: { body: "#e8a486", edge: "#a14e34" },
  drop: { body: "#c4b69a", edge: "#7d705a" },
};
const PRIO = { low: "#8b97ab", medium: "#e0a235", high: "#d6395f" };

function Pkg({ vz, o }) {
  const c = TONE[vz.tone] || TONE.kraft;
  const r = 15;
  const C = 2 * Math.PI * 10.5;
  return (
    <g transform={`translate(${vz.x.toFixed(2)} ${vz.y.toFixed(2)}) scale(${vz.s})`}>
      <rect x={-r / 2} y={-r / 2} width={r} height={r} rx="2.6"
        fill={c.body} stroke={c.edge} strokeWidth="1.1" />
      <rect x="-1.5" y={-r / 2} width="3" height={r} fill="#f3e7cf" opacity="0.9" />
      <rect x={-r / 2} y="-1.5" width={r} height="3" fill="#f3e7cf" opacity="0.9" />
      {o && (
        <circle cx={r / 2 - 3} cy={-r / 2 + 3} r="2.3" fill={PRIO[o.priority]}
          stroke="#fff" strokeWidth="0.6" />
      )}
      {o && o.perishable && (
        <circle cx={-r / 2 + 3.4} cy={r / 2 - 3.4} r="2.6" fill="#eafcef"
          stroke="#2f8f57" strokeWidth="0.9" />
      )}
      {vz.phase === "processing" && (
        <circle r="10.5" fill="none" stroke="#ffb02e" strokeWidth="2.4"
          strokeLinecap="round" strokeDasharray={`${C}`}
          strokeDashoffset={`${C * (1 - (vz.prog || 0))}`} transform="rotate(-90)" />
      )}
      {vz.tone === "unfinished" && (
        <text x="0" y="3.4" textAnchor="middle" fontFamily="'IBM Plex Mono',monospace"
          fontSize="11" fontWeight="700" fill="#7d2a16">!</text>
      )}
    </g>
  );
}

function PickerCell({ i, busy, accent }) {
  const y = G.pickY0 + i * G.pickPitch;
  const cy = y + G.pickPitch / 2 - 1;
  return (
    <g>
      <rect x={G.pickCellX0} y={y + 2} width={G.pickCellX1 - G.pickCellX0}
        height={G.pickPitch - 6} rx="6"
        fill={busy ? "rgba(224,123,53,0.13)" : "rgba(255,255,255,0.4)"}
        stroke={accent} strokeWidth="1" strokeOpacity={busy ? 0.55 : 0.28} />
      <text x={G.pickCellX0 + 13} y={cy + 4} fill={accent}
        fontFamily="'IBM Plex Mono',monospace" fontSize="11" fontWeight="700">
        {`P${i + 1}`}
      </text>
      {/* compact picker figure */}
      <g transform={`translate(${G.pickGlyphX} ${cy})`}>
        <circle cx="0" cy="-7" r="5" fill={busy ? "#e7b78c" : "#c2c8d2"}
          stroke="#2c2c34" strokeWidth="1" />
        <path d="M-8 13 L-7 -2 Q0 -7 7 -2 L8 13 Z" fill={busy ? accent : "#9aa3b2"}
          stroke="#2c2c34" strokeWidth="1" strokeLinejoin="round" />
      </g>
    </g>
  );
}

function TruckMini({ x, y, scale = 1, accent }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${scale})`}>
      <rect x="-42" y="-26" width="52" height="44" rx="4" fill="#eef1f5"
        stroke="#33404f" strokeWidth="1.8" />
      <path d="M10 -2 L10 -24 L28 -24 L38 -6 L38 -2 Z" fill={accent}
        stroke="#33404f" strokeWidth="1.8" strokeLinejoin="round" />
      <rect x="14" y="-20" width="12" height="10" rx="1.6" fill="#bfe0f4"
        stroke="#33404f" strokeWidth="1.4" />
      <circle cx="-26" cy="20" r="8" fill="#2b2b33" stroke="#10131a" strokeWidth="2" />
      <circle cx="24" cy="20" r="8" fill="#2b2b33" stroke="#10131a" strokeWidth="2" />
      <circle cx="-26" cy="20" r="2.8" fill="#aab3c0" />
      <circle cx="24" cy="20" r="2.8" fill="#aab3c0" />
    </g>
  );
}

/* ---------------------------------------------------------------------------
 * 5. Static floor chrome (memoized)
 * ------------------------------------------------------------------------ */
const FloorChrome = React.memo(function FloorChrome({ role }) {
  const isDahs = role === "dahs";
  const sceneTop = 34, sceneBot = 500;
  // orthogonal fan: node -> bus -> 10 picker stubs
  const fan = [];
  fan.push(<line key="trunk" x1={G.node.x} y1={G.node.y} x2={G.fanBusX} y2={G.node.y}
    stroke={PAL.dispatch} strokeWidth="2" strokeOpacity="0.4" />);
  fan.push(<line key="bus" x1={G.fanBusX} y1={pickY(0)} x2={G.fanBusX} y2={pickY(N_PICK - 1)}
    stroke={PAL.dispatch} strokeWidth="2" strokeOpacity="0.4" />);
  for (let i = 0; i < N_PICK; i++) {
    fan.push(<line key={`s${i}`} x1={G.fanBusX} y1={pickY(i)} x2={G.pickInX} y2={pickY(i)}
      stroke={PAL.dispatch} strokeWidth="2" strokeOpacity="0.4" />);
  }
  return (
    <g>
      <rect x="0" y="0" width={G.W} height={G.H} fill="url(#floor)" />
      <g stroke="#d6cab0" strokeWidth="1" opacity="0.5">
        {Array.from({ length: 6 }, (_, i) => (
          <line key={i} x1="0" y1={70 + i * 78} x2={G.W} y2={70 + i * 78} />
        ))}
      </g>

      {/* station zones */}
      {STATIONS.map((s) => (
        <g key={s.key}>
          <rect x={s.x} y={sceneTop} width={s.w} height={sceneBot - sceneTop} rx="11"
            fill="rgba(255,255,255,0.55)" stroke={PAL[s.key]} strokeWidth="1.4"
            strokeOpacity="0.55" />
          <rect x={s.x} y={sceneTop} width={s.w} height="4" rx="2" fill={PAL[s.key]} />
          <circle cx={s.x + 17} cy={sceneTop + 19} r="10" fill={PAL[s.key]} />
          <text x={s.x + 17} y={sceneTop + 23} textAnchor="middle" fill="#fff"
            fontFamily="'IBM Plex Mono',monospace" fontSize="11" fontWeight="700">
            {s.n}
          </text>
          <text x={s.x + 33} y={sceneTop + 23} fill="#2a2620"
            fontFamily="'Source Sans 3',sans-serif" fontSize="13.5" fontWeight="700">
            {s.title}
          </text>
        </g>
      ))}
      {/* station 6 label inside the lower half of the on-time/breach card */}
      <text x={STATIONS[4].x + 33} y={296} fill={PAL.breach}
        fontFamily="'Source Sans 3',sans-serif" fontSize="12.5" fontWeight="700">
        SLA breach / spoilage
      </text>
      <line x1={STATIONS[4].x + 12} y1={284} x2={STATIONS[4].x + STATIONS[4].w - 12}
        y2={284} stroke={PAL.breach} strokeWidth="1" strokeOpacity="0.4"
        strokeDasharray="4 4" />

      {/* inbound dock */}
      <rect x="26" y="196" width="150" height="116" rx="8" fill="#e7decb"
        stroke="#b6a988" strokeWidth="1.6" />
      {Array.from({ length: 4 }, (_, i) => (
        <line key={i} x1="40" y1={214 + i * 20} x2="162" y2={214 + i * 20}
          stroke="#bdb094" strokeWidth="2.4" />
      ))}
      <TruckMini x="118" y="276" scale={0.62} accent={PAL.arrival} />
      <text x="101" y="304" textAnchor="middle" fill="#6b6253"
        fontFamily="'IBM Plex Mono',monospace" fontSize="8.5" fontWeight="600">
        INBOUND
      </text>

      {/* conveyor: dock -> queue (orthogonal) */}
      <line x1={G.spawn.x} y1={G.spawn.y} x2={G.q.x0 - 8} y2={G.spawn.y}
        stroke="#b9ad92" strokeWidth="9" strokeLinecap="round" />
      <line className="dahs-belt" x1={G.spawn.x} y1={G.spawn.y} x2={G.q.x0 - 8}
        y2={G.spawn.y} stroke="#efe7d3" strokeWidth="9" strokeLinecap="round"
        strokeDasharray="2 12" />

      {/* queue capacity frame */}
      <rect x={G.q.x0 - 9} y={G.q.y0 - 9} width={G.q.cols * G.q.pitch + 18}
        height="430" rx="7" fill="none" stroke={PAL.queue} strokeWidth="1.2"
        strokeOpacity="0.4" strokeDasharray="5 5" />
      {/* drop chute */}
      <line x1={G.q.x0 + 60} y1="466" x2={G.dropBin.x} y2="466" stroke="#b9ad92"
        strokeWidth="6" strokeDasharray="2 9" strokeLinecap="round" opacity="0.7" />
      <path d={`M${G.dropBin.x - 22} ${G.dropBin.y - 14} L${G.dropBin.x - 16} ${G.dropBin.y + 16} L${G.dropBin.x + 16} ${G.dropBin.y + 16} L${G.dropBin.x + 22} ${G.dropBin.y - 14} Z`}
        fill="#c9bda1" stroke="#8a7c5e" strokeWidth="1.4" />

      {/* orthogonal flow connectors */}
      <g fill="none" strokeLinecap="round" strokeWidth="3">
        <line x1={190} y1={sceneTop + 13} x2={198} y2={sceneTop + 13}
          stroke={PAL.queue} strokeOpacity="0.6" />
        <line x1={422} y1={G.node.y} x2={430} y2={G.node.y}
          stroke={PAL.dispatch} strokeOpacity="0.6" />
      </g>

      {/* picker fan */}
      {fan}

      {/* feedback loop — DAHS floor only (orthogonal) */}
      {isDahs && (
        <g>
          <polyline points={`${G.node.x},${G.node.y + 18} ${G.node.x},${488} ${G.q.x0 + 70},${488} ${G.q.x0 + 70},${G.q.y0 * 1 + 392}`}
            fill="none" stroke="#c2563a" strokeWidth="2" strokeDasharray="6 5"
            markerEnd="url(#fbArrow)" opacity="0.8" />
          <text x={G.node.x - 150} y={482} fill="#b5472c"
            fontFamily="'IBM Plex Mono',monospace" fontSize="10" fontWeight="700">
            KPI feedback → state
          </text>
        </g>
      )}

      {/* on-time loading dock */}
      <TruckMini x={G.shipDock.x + 96} y={G.shipDock.y - 70} scale={0.6}
        accent={PAL.ontime} />
    </g>
  );
});

/* ---------------------------------------------------------------------------
 * 6. Dispatcher node (per floor, dynamic)
 * ------------------------------------------------------------------------ */
function DispatcherNode({ interval, isDahs, baselineLabel }) {
  const iv = interval || {};
  const rule = iv.rule || HEUR[0];
  const probs = iv.probs;
  const chosen = HEUR.indexOf(rule);
  return (
    <g>
      <circle key={iv.idx} cx={G.node.x} cy={G.node.y} r="30" fill="none"
        stroke={isDahs ? PAL.dispatch : "#8a93a6"} strokeWidth="2.4"
        className="dahs-pulse" />
      <circle cx={G.node.x} cy={G.node.y} r="28"
        fill={isDahs ? "#1f9d57" : "#5d6675"} stroke={isDahs ? "#0e5e33" : "#3c4350"}
        strokeWidth="2.4" />
      <text x={G.node.x} y={G.node.y - 4} textAnchor="middle" fill="#eafff2"
        fontFamily="'IBM Plex Mono',monospace" fontSize="8" fontWeight="600"
        opacity="0.85">
        {isDahs ? "DAHS" : "RULE"}
      </text>
      <text x={G.node.x} y={G.node.y + 11} textAnchor="middle" fill="#fff"
        fontFamily="'IBM Plex Mono',monospace" fontSize="12" fontWeight="700">
        {rule}
      </text>
      {/* probability bars when the policy exposes a distribution */}
      {probs && (
        <g transform={`translate(${G.node.x - 54} ${G.node.y + 42})`}>
          <rect x="-7" y="-10" width="118" height="53" rx="5"
            fill="rgba(255,255,255,0.9)" stroke={isDahs ? PAL.dispatch : "#8a93a6"}
            strokeWidth="1" strokeOpacity="0.5" />
          <text x="0" y="0" fill="#3a4a40" fontFamily="'IBM Plex Mono',monospace"
            fontSize="7" fontWeight="600">P(rule | state)</text>
          {HEUR.map((h, i) => {
            const p = probs[i] || 0;
            const on = i === chosen;
            return (
              <g key={h} transform={`translate(${i * 28} 5)`}>
                <rect x="0" y="0" width="24" height="24" rx="2.5" fill="#e7ede9" />
                <rect x="0" y={24 - p * 24} width="24" height={p * 24} rx="2.5"
                  fill={on ? PAL.dispatch : "#9ab9a8"} />
                <text x="12" y="33" textAnchor="middle"
                  fill={on ? PAL.dispatch : "#6b756f"}
                  fontFamily="'IBM Plex Mono',monospace" fontSize="7"
                  fontWeight={on ? "700" : "500"}>{h}</text>
              </g>
            );
          })}
        </g>
      )}
      {!probs && (
        <text x={G.node.x} y={G.node.y + 52} textAnchor="middle" fill="#6b6253"
          fontFamily="'IBM Plex Mono',monospace" fontSize="8.5">
          {baselineLabel}
        </text>
      )}
    </g>
  );
}

/* ---------------------------------------------------------------------------
 * 7. One floor (dynamic)
 * ------------------------------------------------------------------------ */
function Floor({ floor, t, originY, role, label, sublabel, accent }) {
  const isDahs = role === "dahs";
  const interval = clamp(Math.floor(t / IV_MIN), 0, N_IV - 1);
  const ivData = floor.intervals[interval];

  const { vizList, busy, counts } = useMemo(() => {
    const slotMap = buildSlotMap(floor.O, t);
    const list = [];
    const bz = new Array(N_PICK).fill(false);
    let queue = 0;
    for (const o of floor.O) {
      if (o.outcome !== "dropped" && o.A <= t && (o.start == null || t < o.start)) queue++;
      const vz = orderViz(o, t, slotMap);
      if (!vz) continue;
      list.push({ o, vz });
      if (vz.phase === "processing" || vz.phase === "dispatch") bz[o.picker] = true;
    }
    const rank = { queued: 0, arriving: 1, drop: 1, breachPile: 1, docked: 1,
      processing: 2, outbound: 3, dispatch: 4 };
    list.sort((a, b) => (rank[a.vz.phase] || 0) - (rank[b.vz.phase] || 0));
    return { vizList: list, busy: bz, counts: { queue } };
  }, [floor, t]);

  return (
    <g transform={`translate(0 ${originY})`}>
      {/* floor caption */}
      <rect x="0" y="-2" width={G.W} height="22" fill={isDahs ? "#143b27" : "#3a2f1c"} />
      <circle cx="14" cy="9" r="5" fill={accent} />
      <text x="26" y="13" fill="#f3f0e6" fontFamily="'Source Sans 3',sans-serif"
        fontSize="12.5" fontWeight="700">{label}</text>
      <text x={26 + label.length * 7.4 + 14} y="13" fill="#a9b0bd"
        fontFamily="'IBM Plex Mono',monospace" fontSize="10">{sublabel}</text>
      <text x={G.W - 12} y="13" textAnchor="end" fill="#cdd3dd"
        fontFamily="'IBM Plex Mono',monospace" fontSize="10">
        {`queue ${ivData ? ivData.queueLenEnd : 0}/${QUEUE_CAP}`}
      </text>

      <g transform="translate(0 22)">
        <FloorChrome role={role} />
        <DispatcherNode interval={ivData} isDahs={isDahs} baselineLabel={label} />
        {Array.from({ length: N_PICK }, (_, i) => (
          <PickerCell key={i} i={i} busy={busy[i]} accent={PAL.pickers} />
        ))}
        {vizList.map(({ o, vz }) => <Pkg key={o.id} vz={vz} o={o} />)}
      </g>
    </g>
  );
}

/* ---------------------------------------------------------------------------
 * 8. Live running counts
 * ------------------------------------------------------------------------ */
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
    unfinished: t >= SHIFT_MIN ? O.filter((o) => o.outcome === "unfinished").length : 0,
    throughput: done,
    breachRate: done ? breached / done : 0,
    meanTardy: done ? tardySum / done : 0,
  };
}

/* ---------------------------------------------------------------------------
 * 9. Right panel — scoreboard + DAHS switch log
 * ------------------------------------------------------------------------ */
function Scoreboard({ dahsC, baseC, baseLabel, finalDahs, finalBase }) {
  const rows = [
    { k: "breachRate", label: "SLA breach", fmt: (v) => `${(v * 100).toFixed(1)}%`, lowerWins: true },
    { k: "breached", label: "breached", fmt: (v) => v, lowerWins: true },
    { k: "shipped", label: "shipped on-time", fmt: (v) => v, lowerWins: false },
    { k: "queue", label: "in queue now", fmt: (v) => v, lowerWins: true },
    { k: "throughput", label: "throughput", fmt: (v) => v, lowerWins: false },
    { k: "meanTardy", label: "mean tardiness", fmt: (v) => `${v.toFixed(1)}m`, lowerWins: true },
  ];
  return (
    <div className="dahs-score">
      <div className="dahs-score-head">
        <span className="dahs-score-h dahs-h-dahs">DAHS</span>
        <span className="dahs-score-h dahs-h-metric">metric</span>
        <span className="dahs-score-h dahs-h-base">{baseLabel}</span>
      </div>
      {rows.map((r) => {
        const d = dahsC[r.k], b = baseC[r.k];
        let dWin = false, bWin = false;
        if (d !== b) { (r.lowerWins ? d < b : d > b) ? (dWin = true) : (bWin = true); }
        return (
          <div className="dahs-score-row" key={r.k}>
            <span className={`dahs-score-v ${dWin ? "win" : ""}`}>{r.fmt(d)}</span>
            <span className="dahs-score-m">{r.label}</span>
            <span className={`dahs-score-v ${bWin ? "win" : ""}`}>{r.fmt(b)}</span>
          </div>
        );
      })}
    </div>
  );
}

function SwitchLog({ switchLog, interval }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current?.querySelector(`[data-iv="${interval}"]`);
    if (el) el.scrollIntoView({ block: "nearest" });
  }, [interval]);
  return (
    <div className="dahs-swlog" ref={ref}>
      {switchLog.map((s) => {
        const active = s.idx === interval;
        const cls = `dahs-sw ${active ? "active" : ""} ${s.switched ? "switched" : ""}`;
        return (
          <div className={cls} data-iv={s.idx} key={s.idx}>
            <div className="dahs-sw-top">
              <span className="dahs-sw-iv">{`#${String(s.idx + 1).padStart(2, "0")}`}</span>
              <span className="dahs-sw-t">{fmtClock(s.tStart)}</span>
              <span className="dahs-sw-rule">
                {s.switched && s.fromRule
                  ? <><b className="from">{s.fromRule}</b><i>→</i><b className="to">{s.rule}</b></>
                  : <b className="hold">{s.rule}</b>}
              </span>
            </div>
            {s.reason && <div className="dahs-sw-reason">{s.reason}</div>}
          </div>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * 10. Academic header
 * ------------------------------------------------------------------------ */
function AcademicHeader() {
  return (
    <header className="dahs-hdr">
      <div className="dahs-hdr-rule" />
      <div className="dahs-hdr-main">
        <div className="dahs-hdr-tag">Interactive exhibit</div>
        <h1 className="dahs-hdr-title">
          Rollout-Informed Label-Distribution Learning for Adaptive Heuristic
          Selection in Dynamic Warehouse Dispatching
        </h1>
        <div className="dahs-hdr-sub">
          Discrete-event simulation of one 8-hour order-picking shift ·
          32 decision intervals · 10 pickers · manuscript in preparation,
          <i> Computers &amp; Operations Research</i>
        </div>
      </div>
    </header>
  );
}

/* ---------------------------------------------------------------------------
 * 11. Setup screen
 * ------------------------------------------------------------------------ */
function SetupScreen({ onRun, busy, error, seed, setSeed, baseline, setBaseline }) {
  return (
    <div className="dahs-setup">
      <div className="dahs-setup-card">
        <div className="dahs-setup-h">Configure the simulation run</div>
        <p className="dahs-setup-p">
          DAHS is replayed against one baseline on an <b>identical, seeded order
          stream</b>. Pressing run executes the real Python simulator
          (<code>simulation.warehouse_env</code>) under both policies — nothing is
          fabricated in the browser.
        </p>

        <div className="dahs-field-label">1 · Baseline policy</div>
        <div className="dahs-base-grid">
          {BASELINES.map((b) => (
            <button key={b.key}
              className={`dahs-base ${baseline === b.key ? "sel" : ""}`}
              onClick={() => setBaseline(b.key)}>
              <div className="dahs-base-top">
                <span className="dahs-base-label">{b.label}</span>
                <span className="dahs-base-fam">{b.family}</span>
              </div>
              <div className="dahs-base-src">{b.source}</div>
              <div className="dahs-base-blurb">{b.blurb}</div>
            </button>
          ))}
        </div>

        <div className="dahs-setup-foot">
          <div className="dahs-field">
            <div className="dahs-field-label">2 · Random seed</div>
            <input className="dahs-seed" type="number" min="0" max="4294967295"
              value={seed} onChange={(e) => setSeed(e.target.value)}
              placeholder="e.g. 42" />
            <div className="dahs-seed-note">
              The warehouse order stream is generated from this seed — same seed,
              same shift.
            </div>
          </div>
          <button className="dahs-run" disabled={busy} onClick={onRun}>
            {busy ? "Running simulator…" : "Run simulation ▸"}
          </button>
        </div>
        {error && <div className="dahs-setup-err">{error}</div>}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * 12. App
 * ------------------------------------------------------------------------ */
const SPEEDS = [0.5, 1, 2, 4];
const MIN_PER_SEC = 8;

export default function App() {
  const [phase, setPhase] = useState("setup");      // setup | loading | ready
  const [log, setLog] = useState(null);
  const [error, setError] = useState(null);
  const [seed, setSeed] = useState("42");
  const [baseline, setBaseline] = useState("fifo");
  const [loadMsg, setLoadMsg] = useState("");

  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const tRef = useRef(0);
  const playRef = useRef(false);
  const speedRef = useRef(1);
  const lastRef = useRef(null);
  const endRef = useRef(600);

  // ---- prepared run data --------------------------------------------------
  const prepared = useMemo(() => {
    if (!log) return null;
    const dahsO = prepareFloor(log.dahs.orders);
    const baseO = prepareFloor(log.baseline.orders);
    const maxFinish = [...dahsO, ...baseO].reduce(
      (m, o) => Math.max(m, o.finish || 0), SHIFT_MIN);
    endRef.current = Math.ceil(maxFinish + VT.OUT + VT.LINGER + 12);
    return {
      meta: log.meta,
      dahs: { O: dahsO, intervals: log.dahs.intervals, switchLog: log.dahs.switchLog,
        kpis: log.dahs.kpis, counts: log.dahs.counts },
      baseline: { O: baseO, intervals: log.baseline.intervals,
        switchLog: log.baseline.switchLog, kpis: log.baseline.kpis,
        counts: log.baseline.counts },
    };
  }, [log]);

  // ---- run the real simulator --------------------------------------------
  const runSim = useCallback(async () => {
    const s = parseInt(seed, 10);
    if (!Number.isInteger(s) || s < 0) {
      setError("Enter a non-negative integer seed.");
      return;
    }
    setPhase("loading");
    setError(null);
    setLoadMsg(`Running WarehouseEnv + DAHS + ${baseline} for seed ${s}…`);
    const started = Date.now();
    try {
      const res = await fetch(`/api/run?seed=${s}&baseline=${baseline}`);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      setLog(data);
      tRef.current = 0;
      setT(0);
      playRef.current = true;
      setPlaying(true);
      setPhase("ready");
      setLoadMsg("");
      void started;
    } catch (e) {
      setError(String(e.message || e));
      setPhase("setup");
    }
  }, [seed, baseline]);

  // ---- animation loop -----------------------------------------------------
  useEffect(() => { playRef.current = playing; }, [playing]);
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => {
    let raf;
    function frame(ts) {
      if (lastRef.current == null) lastRef.current = ts;
      const dt = Math.min((ts - lastRef.current) / 1000, 0.1);
      lastRef.current = ts;
      if (playRef.current && phase === "ready") {
        let nt = tRef.current + dt * MIN_PER_SEC * speedRef.current;
        if (nt >= endRef.current) { nt = endRef.current; playRef.current = false; setPlaying(false); }
        tRef.current = nt;
        setT(nt);
      }
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [phase]);

  const setTime = useCallback((v) => { tRef.current = v; setT(v); }, []);
  const onPlay = useCallback(() => {
    if (tRef.current >= endRef.current - 0.5) setTime(0);
    setPlaying((p) => !p);
  }, [setTime]);
  const onSpeed = useCallback(() =>
    setSpeed((s) => SPEEDS[(SPEEDS.indexOf(s) + 1) % SPEEDS.length]), []);

  useEffect(() => {
    const h = (e) => {
      if (e.code === "Space" && phase === "ready") { e.preventDefault(); onPlay(); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onPlay, phase]);

  // ---- render -------------------------------------------------------------
  if (phase !== "ready" || !prepared) {
    return (
      <div className="dahs-app">
        <style>{CSS}</style>
        <AcademicHeader />
        {phase === "loading"
          ? <div className="dahs-loading">
              <div className="dahs-spin" />
              <div className="dahs-load-msg">{loadMsg}</div>
              <div className="dahs-load-sub">
                executing demo/build_run_log.py — real WarehouseEnv, no shortcuts
              </div>
            </div>
          : <SetupScreen onRun={runSim} busy={phase === "loading"} error={error}
              seed={seed} setSeed={setSeed} baseline={baseline}
              setBaseline={setBaseline} />}
      </div>
    );
  }

  const interval = clamp(Math.floor(t / IV_MIN), 0, N_IV - 1);
  const dahsC = liveCounts(prepared.dahs.O, t);
  const baseC = liveCounts(prepared.baseline.O, t);
  const baseMeta = prepared.meta.baseline;
  const atEnd = t >= endRef.current - 0.5;

  return (
    <div className="dahs-app">
      <style>{CSS}</style>
      <AcademicHeader />

      <div className="dahs-body">
        <div className="dahs-stage">
          <svg viewBox={`0 0 ${G.W} 1102`} preserveAspectRatio="xMidYMid meet"
            className="dahs-svg">
            <defs>
              <linearGradient id="floor" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#f6f1e4" />
                <stop offset="1" stopColor="#ebe4d3" />
              </linearGradient>
              <marker id="fbArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7"
                markerHeight="7" orient="auto-start-reverse">
                <path d="M0 0L10 5L0 10z" fill="#c2563a" />
              </marker>
            </defs>
            <rect x="0" y="0" width={G.W} height="1102" fill="#0e1118" />
            <Floor floor={prepared.dahs} t={t} originY={8} role="dahs"
              label="DAHS" sublabel="adaptive rollout-informed selection"
              accent="#37c47f" />
            <Floor floor={prepared.baseline} t={t} originY={558} role="baseline"
              label={baseMeta.label} sublabel={baseMeta.source} accent="#e0a94a" />
          </svg>
        </div>

        <aside className="dahs-side">
          <div className="dahs-side-sec">
            <div className="dahs-side-h">Live comparison · same {prepared.meta.nOrders} orders</div>
            <Scoreboard dahsC={dahsC} baseC={baseC} baseLabel={baseMeta.label} />
          </div>
          <div className="dahs-side-sec dahs-side-grow">
            <div className="dahs-side-h">
              DAHS solver-switching log
              <span className="dahs-side-hh">interval {interval + 1}/{N_IV}</span>
            </div>
            <SwitchLog switchLog={prepared.dahs.switchLog} interval={interval} />
          </div>
        </aside>
      </div>

      <footer className="dahs-foot">
        <button className="dahs-btn dahs-btn-ghost" onClick={() => {
          setPhase("setup"); setPlaying(false); playRef.current = false;
        }}>◂ New run</button>
        <button className="dahs-btn dahs-btn-play" onClick={onPlay}>
          {playing ? "❚❚ Pause" : atEnd ? "↻ Replay" : "▸ Play"}
        </button>
        <button className="dahs-btn" onClick={onSpeed}>{speed}× speed</button>
        <div className="dahs-clock">
          <b>{fmtClock(t)}</b><span> / 08:00 — shift interval {interval + 1}/{N_IV}</span>
        </div>
        <div className="dahs-scrub-wrap">
          <input className="dahs-scrub" type="range" min="0" max={endRef.current}
            step="0.1" value={t} onChange={(e) => setTime(parseFloat(e.target.value))} />
          <div className="dahs-ticks">
            {Array.from({ length: N_IV + 1 }, (_, i) => (
              <span key={i} className="dahs-tick"
                style={{ left: `${(i * IV_MIN / endRef.current) * 100}%` }} />
            ))}
          </div>
        </div>
        <div className="dahs-runtag">
          seed {prepared.meta.seed} · {prepared.meta.arrivalMode} arrivals
        </div>
      </footer>

      {atEnd && (
        <RunSummary dahs={prepared.dahs} base={prepared.baseline}
          meta={prepared.meta} onReplay={() => { setTime(0); setPlaying(true); }}
          onNew={() => { setPhase("setup"); setPlaying(false); playRef.current = false; }} />
      )}
    </div>
  );
}

function RunSummary({ dahs, base, meta, onReplay, onNew }) {
  const k1 = dahs.kpis, k2 = base.kpis;
  const rows = [
    ["SLA breach rate", `${(k1.sla_breach_rate * 100).toFixed(2)}%`,
      `${(k2.sla_breach_rate * 100).toFixed(2)}%`, true],
    ["Spoilage rate", `${(k1.spoilage_rate * 100).toFixed(2)}%`,
      `${(k2.spoilage_rate * 100).toFixed(2)}%`, true],
    ["Mean tardiness", `${k1.mean_tardiness.toFixed(2)}m`,
      `${k2.mean_tardiness.toFixed(2)}m`, true],
    ["Throughput", `${k1.throughput.toFixed(0)}`, `${k2.throughput.toFixed(0)}`, false],
    ["Picker utilisation", `${(k1.picker_utilization * 100).toFixed(1)}%`,
      `${(k2.picker_utilization * 100).toFixed(1)}%`, false],
    ["Unfinished at shift end", `${dahs.counts.unfinished}`,
      `${base.counts.unfinished}`, true],
  ];
  return (
    <div className="dahs-summary">
      <div className="dahs-summary-card">
        <div className="dahs-summary-h">Shift complete — DAHS vs {meta.baseline.label}</div>
        <div className="dahs-summary-sub">
          seed {meta.seed} · {meta.nOrders} orders · KPIs are the real
          <code> simulation.kpis.compute_kpis</code> output of each run.
        </div>
        <table className="dahs-summary-tbl">
          <thead><tr><th>KPI</th><th>DAHS</th><th>{meta.baseline.label}</th></tr></thead>
          <tbody>
            {rows.map(([l, a, b, lowGood]) => {
              const na = parseFloat(a), nb = parseFloat(b);
              const dahsWin = na !== nb && (lowGood ? na < nb : na > nb);
              const baseWin = na !== nb && !dahsWin;
              return (
                <tr key={l}>
                  <td>{l}</td>
                  <td className={dahsWin ? "win" : ""}>{a}</td>
                  <td className={baseWin ? "win" : ""}>{b}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="dahs-summary-btns">
          <button className="dahs-btn dahs-btn-ghost" onClick={onNew}>◂ New run</button>
          <button className="dahs-btn dahs-btn-play" onClick={onReplay}>↻ Replay</button>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * 13. Styles
 * ------------------------------------------------------------------------ */
const CSS = `
.dahs-app{
  height:100%;display:flex;flex-direction:column;background:#0e1118;
  color:#e6e9f0;font-family:'Source Sans 3',sans-serif;overflow:hidden;
}
/* ---- academic header ---- */
.dahs-hdr{flex:0 0 auto;background:#141821;border-bottom:1px solid #283040;
  padding:10px 26px 12px;position:relative;}
.dahs-hdr-rule{position:absolute;left:0;top:0;height:3px;width:100%;
  background:linear-gradient(90deg,#3b6fc9,#7c5bd0,#1f9d57,#e07b35,#d6483c);}
.dahs-hdr-tag{font-family:'IBM Plex Mono',monospace;font-size:10px;
  letter-spacing:2.5px;text-transform:uppercase;color:#7f8aa0;margin-bottom:3px;}
.dahs-hdr-title{font-family:'Playfair Display',serif;font-weight:600;
  font-size:21px;line-height:1.25;color:#f1ede3;margin:0;max-width:1180px;}
.dahs-hdr-sub{font-size:12px;color:#9aa3b6;margin-top:4px;}
.dahs-hdr-sub i{color:#c3b9a4;}
/* ---- setup ---- */
.dahs-setup{flex:1;display:flex;align-items:flex-start;justify-content:center;
  overflow:auto;padding:26px;}
.dahs-setup-card{width:920px;max-width:100%;background:#161b25;
  border:1px solid #2a3240;border-radius:14px;padding:24px 28px;}
.dahs-setup-h{font-family:'Playfair Display',serif;font-size:22px;font-weight:600;
  color:#f1ede3;}
.dahs-setup-p{font-size:13px;color:#9aa3b6;line-height:1.6;margin:6px 0 18px;}
.dahs-setup-p code,.dahs-summary-sub code{font-family:'IBM Plex Mono',monospace;
  font-size:11.5px;color:#cdd3dd;background:#0c1019;padding:1px 5px;border-radius:4px;}
.dahs-field-label{font-family:'IBM Plex Mono',monospace;font-size:11px;
  letter-spacing:1.5px;text-transform:uppercase;color:#8893a8;margin-bottom:9px;}
.dahs-base-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;
  margin-bottom:20px;}
.dahs-base{text-align:left;background:#10141d;border:1px solid #2a3240;
  border-radius:9px;padding:10px 13px;cursor:pointer;transition:.14s;color:inherit;}
.dahs-base:hover{border-color:#46557a;background:#141a26;}
.dahs-base.sel{border-color:#37c47f;background:#13251c;
  box-shadow:0 0 0 1px #37c47f inset;}
.dahs-base-top{display:flex;align-items:baseline;gap:9px;}
.dahs-base-label{font-family:'IBM Plex Mono',monospace;font-size:15px;
  font-weight:700;color:#f1ede3;}
.dahs-base-fam{font-size:10.5px;color:#7f8aa0;text-transform:uppercase;
  letter-spacing:.6px;}
.dahs-base-src{font-size:11.5px;color:#b7a98e;margin-top:3px;}
.dahs-base-blurb{font-size:11.5px;color:#8a93a8;margin-top:2px;line-height:1.4;}
.dahs-setup-foot{display:flex;align-items:flex-end;gap:22px;}
.dahs-field{flex:1;}
.dahs-seed{width:100%;background:#10141d;border:1px solid #2a3240;border-radius:8px;
  padding:9px 12px;color:#f1ede3;font-family:'IBM Plex Mono',monospace;font-size:15px;}
.dahs-seed:focus{outline:none;border-color:#37c47f;}
.dahs-seed-note{font-size:11px;color:#7f8aa0;margin-top:5px;}
.dahs-run{background:linear-gradient(135deg,#37c47f,#1f9d57);color:#06140c;
  border:none;border-radius:9px;font-weight:700;font-size:15px;padding:12px 24px;
  cursor:pointer;font-family:'Source Sans 3',sans-serif;white-space:nowrap;}
.dahs-run:hover{filter:brightness(1.08);}
.dahs-run:disabled{opacity:.6;cursor:default;}
.dahs-setup-err{margin-top:14px;background:#2a1518;border:1px solid #6b2b2b;
  border-radius:8px;padding:10px 13px;color:#f0a9a0;font-size:12.5px;
  white-space:pre-wrap;font-family:'IBM Plex Mono',monospace;}
/* ---- loading ---- */
.dahs-loading{flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:14px;}
.dahs-spin{width:46px;height:46px;border-radius:50%;border:4px solid #243150;
  border-top-color:#37c47f;animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.dahs-load-msg{font-size:14px;color:#dbe0ea;font-weight:600;}
.dahs-load-sub{font-size:11.5px;color:#7f8aa0;font-family:'IBM Plex Mono',monospace;}
/* ---- body ---- */
.dahs-body{flex:1;display:flex;min-height:0;}
.dahs-stage{flex:1;min-width:0;display:flex;background:#0e1118;}
.dahs-svg{width:100%;height:100%;display:block;}
.dahs-side{flex:0 0 332px;background:#141821;border-left:1px solid #283040;
  display:flex;flex-direction:column;min-height:0;}
.dahs-side-sec{border-bottom:1px solid #283040;padding:11px 14px;}
.dahs-side-grow{flex:1;min-height:0;display:flex;flex-direction:column;}
.dahs-side-h{font-family:'IBM Plex Mono',monospace;font-size:10.5px;
  letter-spacing:1px;text-transform:uppercase;color:#8893a8;margin-bottom:9px;
  display:flex;justify-content:space-between;}
.dahs-side-hh{color:#37c47f;}
/* ---- scoreboard ---- */
.dahs-score-head,.dahs-score-row{display:grid;
  grid-template-columns:1fr 1.25fr 1fr;align-items:center;}
.dahs-score-head{margin-bottom:5px;}
.dahs-score-h{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:.5px;text-align:center;}
.dahs-h-dahs{color:#37c47f;}.dahs-h-base{color:#e0a94a;}.dahs-h-metric{color:#6b7488;}
.dahs-score-row{padding:4px 0;border-top:1px solid #222a38;}
.dahs-score-v{font-family:'IBM Plex Mono',monospace;font-size:15px;font-weight:700;
  text-align:center;color:#aeb6c6;}
.dahs-score-v.win{color:#fff;}
.dahs-score-row:has(.dahs-score-v.win:first-child) .dahs-score-v:first-child{}
.dahs-score-v.win{position:relative;}
.dahs-score-m{font-size:11px;color:#8893a8;text-align:center;}
/* ---- switch log ---- */
.dahs-swlog{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:4px;
  padding-right:3px;}
.dahs-sw{background:#10141d;border:1px solid #232b3a;border-left:3px solid #2f3a4d;
  border-radius:6px;padding:5px 8px;}
.dahs-sw.switched{border-left-color:#e07b35;}
.dahs-sw.active{background:#172234;border-color:#37c47f;}
.dahs-sw-top{display:flex;align-items:center;gap:7px;}
.dahs-sw-iv{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#6b7488;}
.dahs-sw-t{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:#8893a8;}
.dahs-sw-rule{margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:11px;}
.dahs-sw-rule i{color:#6b7488;margin:0 3px;font-style:normal;}
.dahs-sw-rule b{font-weight:700;}
.dahs-sw-rule .from{color:#8893a8;}.dahs-sw-rule .to{color:#f0a55c;}
.dahs-sw-rule .hold{color:#aeb6c6;}
.dahs-sw-reason{font-size:10.5px;color:#7c879c;margin-top:2px;line-height:1.35;}
.dahs-sw.active .dahs-sw-reason{color:#a9b6cc;}
/* ---- footer ---- */
.dahs-foot{flex:0 0 auto;display:flex;align-items:center;gap:12px;
  padding:9px 18px;background:#141821;border-top:1px solid #283040;}
.dahs-btn{background:#1c2433;border:1px solid #313c52;color:#dbe0ea;
  border-radius:8px;font-family:'Source Sans 3',sans-serif;font-weight:600;
  font-size:13px;padding:8px 13px;cursor:pointer;white-space:nowrap;}
.dahs-btn:hover{background:#243150;}
.dahs-btn-play{background:linear-gradient(135deg,#37c47f,#1f9d57);color:#06140c;
  border:none;font-weight:700;min-width:96px;}
.dahs-btn-ghost{background:transparent;}
.dahs-clock{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#8893a8;
  white-space:nowrap;}
.dahs-clock b{color:#f1ede3;font-size:16px;}
.dahs-scrub-wrap{flex:1;position:relative;min-width:120px;}
.dahs-scrub{-webkit-appearance:none;appearance:none;width:100%;height:6px;
  border-radius:4px;background:linear-gradient(90deg,#3b6fc9,#7c5bd0,#1f9d57,#e07b35,#d6483c);
  outline:none;cursor:pointer;}
.dahs-scrub::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
  width:16px;height:16px;border-radius:50%;background:#fff;border:3px solid #37c47f;
  cursor:pointer;}
.dahs-scrub::-moz-range-thumb{width:16px;height:16px;border-radius:50%;
  background:#fff;border:3px solid #37c47f;cursor:pointer;}
.dahs-ticks{position:absolute;left:0;right:0;top:11px;height:5px;
  pointer-events:none;}
.dahs-tick{position:absolute;width:1px;height:4px;background:rgba(255,255,255,0.3);}
.dahs-runtag{font-family:'IBM Plex Mono',monospace;font-size:10.5px;
  color:#6b7488;white-space:nowrap;}
/* ---- animations ---- */
.dahs-belt{animation:belt 1.1s linear infinite;}
@keyframes belt{to{stroke-dashoffset:-28;}}
.dahs-pulse{transform-origin:center;transform-box:fill-box;
  animation:pulse 1.5s ease-out forwards;}
@keyframes pulse{from{opacity:.6;transform:scale(.7);}to{opacity:0;transform:scale(1.9);}}
/* ---- summary ---- */
.dahs-summary{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;background:rgba(8,10,16,0.8);backdrop-filter:blur(3px);
  animation:fade .3s ease;}
@keyframes fade{from{opacity:0;}to{opacity:1;}}
.dahs-summary-card{background:#161b25;border:1px solid #2d3a4e;border-radius:14px;
  padding:24px 30px;width:540px;box-shadow:0 30px 80px rgba(0,0,0,.6);}
.dahs-summary-h{font-family:'Playfair Display',serif;font-size:22px;font-weight:600;
  color:#f1ede3;}
.dahs-summary-sub{font-size:12px;color:#9aa3b6;margin:5px 0 16px;line-height:1.5;}
.dahs-summary-tbl{width:100%;border-collapse:collapse;margin-bottom:18px;}
.dahs-summary-tbl th,.dahs-summary-tbl td{padding:7px 10px;text-align:right;
  font-size:13px;border-bottom:1px solid #232b3a;}
.dahs-summary-tbl th{font-family:'IBM Plex Mono',monospace;font-size:10.5px;
  text-transform:uppercase;letter-spacing:.5px;color:#8893a8;}
.dahs-summary-tbl th:first-child,.dahs-summary-tbl td:first-child{text-align:left;
  color:#aeb6c6;}
.dahs-summary-tbl td{font-family:'IBM Plex Mono',monospace;color:#cdd3dd;}
.dahs-summary-tbl td.win{color:#fff;font-weight:700;background:#13251c;}
.dahs-summary-btns{display:flex;gap:10px;justify-content:flex-end;}
`;
