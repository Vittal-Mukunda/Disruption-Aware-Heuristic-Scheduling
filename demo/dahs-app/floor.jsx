/* DAHS Dashboard — floor diagram.
 *  10 pickers stacked vertically. Orthogonal fan from the DAHS brain at left
 *  to each picker; convergence bus on the right collapses 10 output streams
 *  into 2 lines (Shipped above, SLA-breach below). The DAHS floor draws a
 *  prominent animated feedback loop from outbound KPIs back to the brain. */

const G = (() => {
  const W = 1320,H = 600;
  const pickRowH = 50,pickY0 = 40,pickBoxX = 580,pickBoxW = 220,pickBoxH = 42;
  return {
    W, H,
    st: [
    { key: "arrival", n: 1, title: "Order arrival", x: 14, w: 156 },
    { key: "queue", n: 2, title: "Order queue", x: 182, w: 220 },
    { key: "dispatch", n: 3, title: "Dispatcher", x: 414, w: 116 },
    { key: "pickers", n: 4, title: "Picker pool", x: 558, w: 248 },
    { key: "outbound", n: 5, title: "Outbound", x: 870, w: 436 }],

    /* spawn point sits at the truck back door so packages emerge from the truck */
    spawn: { x: 110, y: 300 },
    truck: { x: 86, y: 300 },
    qGrid: { x: 190, y: 48, cols: 11, pitch: 17 },
    qCap: 200,
    node: { x: 472, y: 300 },
    pickRowH, pickY0, pickN: 10,
    pickBoxX, pickBoxW, pickBoxH,
    pickInX: pickBoxX,
    pickOutX: pickBoxX + pickBoxW,
    fanBusX: 552,
    convBusX: 818,
    shipExitY: 170,
    breachExitY: 450,
    shipPile: { x: 894, y: 80, cols: 8, pitch: 18 },
    shipDock: { x: 1258, y: 170 },
    breachPile: { x: 894, y: 360, cols: 8, pitch: 17 }
  };
})();

const pickerCenterY = (i) => G.pickY0 + i * G.pickRowH + G.pickBoxH / 2;
const qSlot = (k) => ({
  x: G.qGrid.x + k % G.qGrid.cols * G.qGrid.pitch + G.qGrid.pitch / 2,
  y: G.qGrid.y + Math.floor(k / G.qGrid.cols) * G.qGrid.pitch + G.qGrid.pitch / 2
});
const gridPos = (g, i) => ({
  x: g.x + i % g.cols * g.pitch + g.pitch / 2,
  y: g.y + Math.floor(i / g.cols) * g.pitch + g.pitch / 2
});

/* ────────── per-order viz at time t ────────── */
function orderViz(o, t, slotMap) {
  const S = window.DAHS_SIM;
  if (t < o.A) return null;

  if (o.outcome === "dropped") {
    const t1 = o.A + S.VT.ARRIVE;
    if (t < t1) {
      const u = S.ease((t - o.A) / S.VT.ARRIVE);
      const p = S.ptOnPolyline([G.spawn, { x: 350, y: G.spawn.y }], u);
      return { ...p, phase: "arriving", tone: "kraft" };
    }
    return null;
  }

  // ARRIVING — spawn → conveyor → queue slot.
  if (t < o.enterQ) {
    const u = S.ease((t - o.A) / Math.max(o.enterQ - o.A, 1e-6));
    const slot = qSlot(o.entrySlot);
    const p = S.ptOnPolyline(
      [G.spawn, { x: slot.x, y: G.spawn.y }, slot], u);
    return { ...p, phase: "arriving", tone: "kraft" };
  }

  // QUEUED — in current slot
  if (t < o.leaveQ) {
    const k = slotMap.get(o.id) ?? o.entrySlot ?? 0;
    const p = qSlot(k);
    return { x: p.x, y: p.y + Math.sin(t * 1.1 + o.id * 0.7) * 0.6,
      phase: "queued", tone: "kraft" };
  }

  // DISPATCH — exit queue → through dispatcher → fan bus → into picker
  if (o.start != null && t < o.start) {
    const u = S.ease((t - o.leaveQ) / Math.max(o.start - o.leaveQ, 1e-6));
    const exit = qSlot(o.exitSlot);
    const py = pickerCenterY(o.picker);
    const path = [
    exit,
    { x: exit.x, y: G.node.y },
    { x: G.node.x, y: G.node.y },
    { x: G.fanBusX, y: G.node.y },
    { x: G.fanBusX, y: py },
    { x: G.pickInX + 22, y: py }];

    const p = S.ptOnPolyline(path, u);
    return { ...p, phase: "dispatch", tone: "active" };
  }

  // PROCESSING — inside picker box (circular progress)
  if (o.start != null && t < o.finish) {
    return {
      x: G.pickBoxX + G.pickBoxW * 0.45,
      y: pickerCenterY(o.picker),
      phase: "processing",
      prog: S.clamp((t - o.start) / Math.max(o.finish - o.start, 1e-6), 0, 1),
      tone: "active"
    };
  }

  // OUTBOUND — out of picker → converge bus → ship-line or breach-line → pile
  if (o.start != null) {
    const py = pickerCenterY(o.picker);
    const breached = o.outcome === "breached";
    const exitY = breached ? G.breachExitY : G.shipExitY;
    const pile = breached ?
    gridPos(G.breachPile, o.gi) :
    gridPos(G.shipPile, o.gi % 40);
    const tOut = o.finish + S.VT.OUT;
    if (t < tOut) {
      const u = S.ease((t - o.finish) / S.VT.OUT);
      const path = [
      { x: G.pickOutX - 20, y: py },
      { x: G.convBusX, y: py },
      { x: G.convBusX, y: exitY },
      { x: pile.x, y: exitY },
      pile];

      const p = S.ptOnPolyline(path, u);
      return { ...p, phase: "outbound",
        tone: o.spoiled ? "spoiled" : breached ? "breach" : "ship" };
    }
    if (breached) {
      return { ...pile, phase: "breachPile",
        tone: o.spoiled ? "spoiled" : "breach" };
    }
    if (t < tOut + S.VT.LINGER) {
      // First ~35% of LINGER: package rests at the pile slot. Remaining 65%:
      // it animates along the SHIPPED line into the back of the truck —
      // visualising the pickup that closes the journey.
      const frac = (t - tOut) / S.VT.LINGER;
      if (frac < 0.35) {
        return { ...pile, phase: "docked", tone: "ship" };
      }
      const truckBack = { x: G.shipDock.x - 30, y: G.shipExitY };
      const u = S.ease((frac - 0.35) / 0.65);
      const p = S.ptOnPolyline([
        pile,
        { x: pile.x, y: G.shipExitY },
        truckBack,
      ], u);
      return { ...p, phase: "loading", tone: "ship" };
    }
    return null;
  }
  return null;
}

/* ────────── glyphs ────────── */
const TONE = {
  kraft: { body: "#dcc59a", edge: "#8a6e35" },
  active: { body: "#e8c283", edge: "#a06b27" },
  ship: { body: "#9fcfb1", edge: "#2c7e58" },
  breach: { body: "#e3a896", edge: "#aa3b2c" },
  spoiled: { body: "#a8a376", edge: "#5e572f" },
  unfinished: { body: "#e8a486", edge: "#8a3b1f" }
};
const PRIO = { low: "#8b97ab", medium: "#c89238", high: "#aa3b2c" };

function Pkg({ vz, o }) {
  const c = TONE[vz.tone] || TONE.kraft;
  const r = 13;
  const C = 2 * Math.PI * 9;
  return (
    <g transform={`translate(${vz.x.toFixed(2)} ${vz.y.toFixed(2)})`}>
      <rect x={-r / 2} y={-r / 2} width={r} height={r}
      fill={c.body} stroke={c.edge} strokeWidth="0.9" />
      <line x1="0" y1={-r / 2} x2="0" y2={r / 2} stroke="#fff8e3" strokeWidth="0.6" opacity="0.85" />
      <line x1={-r / 2} y1="0" x2={r / 2} y2="0" stroke="#fff8e3" strokeWidth="0.6" opacity="0.85" />
      {o &&
      <circle cx={r / 2 - 2.5} cy={-r / 2 + 2.5} r="2"
      fill={PRIO[o.priority] || PRIO.medium}
      stroke="#fff" strokeWidth="0.5" />
      }
      {o && o.perishable &&
      <circle cx={-r / 2 + 3} cy={r / 2 - 3} r="2.2"
      fill="#eafcef" stroke="#2c7e58" strokeWidth="0.7" />
      }
      {vz.phase === "processing" &&
      <circle r="9" fill="none" stroke="#a06b27" strokeWidth="2"
      strokeLinecap="round" strokeDasharray={`${C}`}
      strokeDashoffset={`${C * (1 - (vz.prog || 0))}`}
      transform="rotate(-90)" />
      }
    </g>);

}

/* ────────── animated inbound truck ────────── */
function InboundTruck({ x, y }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      {/* dock pad behind truck */}
      <rect x="-78" y="-104" width="156" height="208" rx="3"
      fill="#ebd9b3" stroke="#9a7a3a" strokeWidth="0.8" />
      {Array.from({ length: 14 }).map((_, i) =>
      <line key={i} x1="-72" y1={-96 + i * 14} x2="72" y2={-96 + i * 14}
      stroke="#bba474" strokeWidth="1.4" />
      )}
      <text x="0" y="-112" textAnchor="middle"
      fontFamily="'IBM Plex Mono', monospace" fontSize="9"
      fill="rgba(28,24,20,0.6)" letterSpacing="2px">RECEIVING DOCK</text>

      {/* truck body (subtle bounce) */}
      <g>
        <animateTransform attributeName="transform" type="translate"
        values="0 0; 0 -0.6; 0 0" dur="0.9s" repeatCount="indefinite" />
        {/* trailer */}
        <rect x="-58" y="-22" width="60" height="44" rx="2"
        fill="#f7eed2" stroke="#1c1814" strokeWidth="1.1" />
        {/* cargo door slats */}
        <line x1="-46" y1="-18" x2="-46" y2="18" stroke="#1c1814" strokeWidth="0.5" opacity="0.45" />
        <line x1="-34" y1="-18" x2="-34" y2="18" stroke="#1c1814" strokeWidth="0.5" opacity="0.45" />
        <line x1="-22" y1="-18" x2="-22" y2="18" stroke="#1c1814" strokeWidth="0.5" opacity="0.45" />
        <line x1="-10" y1="-18" x2="-10" y2="18" stroke="#1c1814" strokeWidth="0.5" opacity="0.45" />
        {/* small reflectors on side */}
        <rect x="-54" y="-20" width="2" height="3" fill="#aa3b2c" />
        <rect x="-2" y="-20" width="2" height="3" fill="#aa3b2c" />
        {/* cab */}
        <path d="M 2 -16 L 2 16 L 36 16 L 44 2 L 44 -10 L 28 -16 Z"
        fill="#e0d7b4" stroke="#1c1814" strokeWidth="1.1"
        strokeLinejoin="round" />
        {/* windshield */}
        <path d="M 18 -12 L 26 -12 L 38 -4 L 38 0 L 18 0 Z"
        fill="#cfe6f1" stroke="#1c1814" strokeWidth="0.7" />
        {/* headlight */}
        <circle cx="43" cy="3" r="1.6" fill="#fff5c5" stroke="#1c1814" strokeWidth="0.4" />
        {/* door handle */}
        <rect x="6" y="2" width="2" height="6" fill="#1c1814" />
      </g>

      {/* wheels (rotate) — back, middle, front */}
      <g transform="translate(-44 22)">
        <circle r="7" fill="#1c1814" />
        <circle r="3" fill="#5a5040" />
        <g>
          <animateTransform attributeName="transform" type="rotate"
          from="0" to="360" dur="1.1s" repeatCount="indefinite" />
          <line x1="-7" y1="0" x2="7" y2="0" stroke="#aab3c0" strokeWidth="0.8" />
          <line x1="0" y1="-7" x2="0" y2="7" stroke="#aab3c0" strokeWidth="0.8" />
          <line x1="-5" y1="-5" x2="5" y2="5" stroke="#8a93a0" strokeWidth="0.5" />
          <line x1="-5" y1="5" x2="5" y2="-5" stroke="#8a93a0" strokeWidth="0.5" />
        </g>
      </g>
      <g transform="translate(-20 22)">
        <circle r="7" fill="#1c1814" />
        <circle r="3" fill="#5a5040" />
        <g>
          <animateTransform attributeName="transform" type="rotate"
          from="0" to="360" dur="1.1s" begin="-0.3s" repeatCount="indefinite" />
          <line x1="-7" y1="0" x2="7" y2="0" stroke="#aab3c0" strokeWidth="0.8" />
          <line x1="0" y1="-7" x2="0" y2="7" stroke="#aab3c0" strokeWidth="0.8" />
        </g>
      </g>
      <g transform="translate(28 22)">
        <circle r="7" fill="#1c1814" />
        <circle r="3" fill="#5a5040" />
        <g>
          <animateTransform attributeName="transform" type="rotate"
          from="0" to="360" dur="1.1s" begin="-0.6s" repeatCount="indefinite" />
          <line x1="-7" y1="0" x2="7" y2="0" stroke="#aab3c0" strokeWidth="0.8" />
          <line x1="0" y1="-7" x2="0" y2="7" stroke="#aab3c0" strokeWidth="0.8" />
        </g>
      </g>

      {/* exhaust pipe + puffs (above cab) */}
      <rect x="30" y="-22" width="3" height="6" fill="#5a5040" />
      {[0, 1, 2].map((i) =>
      <circle key={i} cx="31.5" cy="-24" r="3" fill="#b9b3a4" opacity="0">
          <animate attributeName="cy"
        values="-24;-48" dur="1.8s" begin={`${i * 0.6}s`} repeatCount="indefinite" />
          <animate attributeName="cx"
        values="31.5;26" dur="1.8s" begin={`${i * 0.6}s`} repeatCount="indefinite" />
          <animate attributeName="r"
        values="2;7" dur="1.8s" begin={`${i * 0.6}s`} repeatCount="indefinite" />
          <animate attributeName="opacity"
        values="0;0.5;0" dur="1.8s" begin={`${i * 0.6}s`} repeatCount="indefinite" />
        </circle>
      )}

      {/* waiting truck (silhouette, behind) */}
      <g transform="translate(0 -76)" opacity="0.32">
        <rect x="-46" y="-14" width="50" height="28" fill="#c7baa0" stroke="#1c1814" strokeWidth="0.7" />
        <path d="M 4 -10 L 4 10 L 26 10 L 30 0 L 30 -6 L 18 -10 Z"
        fill="#bfb29a" stroke="#1c1814" strokeWidth="0.7"
        strokeLinejoin="round" />
        <circle cx="-32" cy="16" r="4" fill="#1c1814" />
        <circle cx="20" cy="16" r="4" fill="#1c1814" />
        <text x="-12" y="-22" textAnchor="middle"
        fontFamily="'IBM Plex Mono', monospace" fontSize="7"
        fill="rgba(28,24,20,0.6)" letterSpacing="1px">NEXT IN LINE</text>
      </g>
    </g>);

}

/* ────────── animated conveyor belt ────────── */
function ConveyorBelt({ x1, y, x2 }) {
  return (
    <g>
      {/* belt body */}
      <rect x={x1} y={y - 6} width={x2 - x1} height={12}
      fill="#c8baa0" stroke="#9a7a3a" strokeWidth="0.7" />
      {/* dark belt edge */}
      <line x1={x1} y1={y - 6} x2={x2} y2={y - 6} stroke="#9a7a3a" strokeWidth="0.7" />
      <line x1={x1} y1={y + 6} x2={x2} y2={y + 6} stroke="#9a7a3a" strokeWidth="0.7" />
      {/* moving treads */}
      <line x1={x1} y1={y} x2={x2} y2={y}
      stroke="#3b322a" strokeWidth="10" strokeLinecap="butt"
      strokeDasharray="4 8" opacity="0.18">
        <animate attributeName="stroke-dashoffset"
        from="0" to="-24" dur="1.0s" repeatCount="indefinite" />
      </line>
      <line x1={x1} y1={y} x2={x2} y2={y}
      stroke="#fff" strokeWidth="2"
      strokeDasharray="2 14" opacity="0.55">
        <animate attributeName="stroke-dashoffset"
        from="0" to="-16" dur="0.7s" repeatCount="indefinite" />
      </line>
      {/* rollers (start and end pulleys) */}
      <circle cx={x1} cy={y} r="7" fill="#5a4f3e" stroke="#1c1814" strokeWidth="0.7" />
      <circle cx={x2} cy={y} r="7" fill="#5a4f3e" stroke="#1c1814" strokeWidth="0.7" />
    </g>);

}

/* ────────── animated feedback loop (DAHS only) ──────────
 * Loops UNDER the floor: out from the outbound exits, around the bottom of
 * the diagram, up the left side, and into the dispatcher from the left.
 * Stays clear of the station headings at the top. Animated traveling
 * dashes + pulsing label make the closed-loop nature visible at a glance. */
function FeedbackLoop() {
  const aX = G.convBusX, aY = (G.shipExitY + G.breachExitY) / 2;
  const bX = G.node.x - 32, bY = G.node.y;       // enters dispatcher from LEFT
  const lowY  = G.H - 22;
  const leftX = G.qGrid.x - 52;                  // up along left side of queue

  const d = `
    M ${aX} ${aY}
    L ${aX + 36} ${aY}
    Q ${aX + 56} ${aY} ${aX + 56} ${aY + 20}
    L ${aX + 56} ${lowY - 20}
    Q ${aX + 56} ${lowY} ${aX + 36} ${lowY}
    L ${leftX + 20} ${lowY}
    Q ${leftX} ${lowY} ${leftX} ${lowY - 20}
    L ${leftX} ${bY + 20}
    Q ${leftX} ${bY} ${leftX + 20} ${bY}
    L ${bX} ${bY}
  `;
  return (
    <g>
      <defs>
        <marker id="fb-arrow-big" viewBox="0 0 12 12" refX="10" refY="6"
        markerWidth="10" markerHeight="10" orient="auto-start-reverse">
          <path d="M0 0 L12 6 L0 12 L3 6 Z" fill="#aa3b2c" />
        </marker>
      </defs>

      {/* glow halo */}
      <path d={d} fill="none" stroke="#aa3b2c" strokeWidth="6.5"
      opacity="0.10" strokeLinecap="round" strokeLinejoin="round" />
      {/* base track */}
      <path d={d} fill="none" stroke="#aa3b2c" strokeWidth="1.6"
      opacity="0.35" strokeLinecap="round" strokeLinejoin="round" />
      {/* travelling dashes — the data flowing back */}
      <path d={d} fill="none" stroke="#aa3b2c" strokeWidth="2.6"
      strokeLinecap="round" strokeLinejoin="round"
      strokeDasharray="14 12" markerEnd="url(#fb-arrow-big)">
        <animate attributeName="stroke-dashoffset"
        from="0" to="-52" dur="1.6s" repeatCount="indefinite" />
      </path>

      {/* small data pulses travelling along the path */}
      {[0, 0.33, 0.66].map((off, i) =>
      <circle key={i} r="3.4" fill="#aa3b2c">
          <animateMotion dur="3.6s" begin={`${-off * 3.6}s`} repeatCount="indefinite"
        keyPoints={`1;0`} keyTimes="0;1">
            <mpath href="#fb-path" />
          </animateMotion>
        </circle>
      )}
      <path id="fb-path" d={d} fill="none" stroke="none" />

      {/* feedback label — in the bottom horizontal stretch, centered */}
      <g transform={`translate(${(aX + leftX) / 2 + 30} ${lowY - 11})`}>
        <rect x="-138" y="-15" width="276" height="22"
        fill="#fbf7e8" stroke="#aa3b2c" strokeWidth="0.8" rx="3" />
        <text x="0" y="0" textAnchor="middle"
        fontFamily="'Playfair Display', serif" fontSize="13"
        fontStyle="italic" fontWeight="600" fill="#aa3b2c">
          KPI feedback → DAHS state
          <animate attributeName="opacity"
          values="0.85;1;0.85" dur="2.4s" repeatCount="indefinite" />
        </text>
      </g>

      {/* annotation into the brain — left side */}
      <g transform={`translate(${bX - 4} ${bY - 6})`}>
        <text x="0" y="0" fontFamily="'IBM Plex Mono', monospace"
        fontSize="9" fill="#aa3b2c" letterSpacing="0.5px"
        textAnchor="end">consumes</text>
      </g>
    </g>);

}

/* ────────── floor chrome (static + animated) ────────── */
function FloorChrome({ role }) {
  const isDahs = role === "dahs";
  const accent = isDahs ? "#216845" : "#a05b1f";

  return (
    <g>
      {/* background — cream paper */}
      <rect width={G.W} height={G.H} fill="#fbf7e8" />

      {/* faint grid */}
      <defs>
        <pattern id={`floor-grid-${role}`} width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M20 0 H0 V20" fill="none" stroke="rgba(28,24,20,0.05)" strokeWidth="0.5" />
        </pattern>
      </defs>
      <rect width={G.W} height={G.H} fill={`url(#floor-grid-${role})`} />

      {/* station bands */}
      {G.st.map((s) =>
      <g key={s.key}>
          <rect x={s.x} y={14} width={s.w} height={G.H - 28}
        fill="none" stroke="rgba(28,24,20,0.18)" strokeWidth="0.7"
        strokeDasharray="3 3" />
          <circle cx={s.x + 12} cy={6} r="8.5" fill="#1c1814" />
          <text x={s.x + 12} y={9.5} textAnchor="middle"
        fontFamily="'IBM Plex Mono', monospace" fontSize="10"
        fontWeight="700" fill="#faf5e8">{s.n}</text>
          <text x={s.x + 26} y={9.5} fill="#1c1814"
        fontFamily="'Playfair Display', serif"
        fontSize="11.5" fontWeight="600">{s.title}</text>
        </g>
      )}

      {/* arrival: dock + animated truck */}
      <InboundTruck x={G.truck.x} y={G.truck.y} />

      {/* conveyor belt: truck back → queue entrance */}
      <ConveyorBelt x1={G.spawn.x + 4} x2={G.qGrid.x - 8} y={G.spawn.y} />

      {/* queue cap frame */}
      <rect x={G.qGrid.x - 8} y={G.qGrid.y - 8}
      width={G.qGrid.cols * G.qGrid.pitch + 16}
      height={G.H - 70}
      fill="none" stroke="#5f4694" strokeWidth="0.8"
      strokeOpacity="0.45" strokeDasharray="5 4" rx="3" />
      <text x={G.qGrid.x + G.qGrid.cols * G.qGrid.pitch + 4} y={G.H - 24}
      textAnchor="end" fontFamily="'IBM Plex Mono', monospace"
      fontSize="9" fill="rgba(95,70,148,0.7)" letterSpacing="1px">
        QUEUE CAP {G.qCap}
      </text>
      {/* faint slot guides inside the queue frame so it never looks empty */}
      {Array.from({ length: 14 }).map((_, row) =>
      Array.from({ length: G.qGrid.cols }).map((__, col) => {
        const cx = G.qGrid.x + col * G.qGrid.pitch + G.qGrid.pitch / 2;
        const cy = G.qGrid.y + row * G.qGrid.pitch + G.qGrid.pitch / 2;
        return <circle key={`${row}-${col}`} cx={cx} cy={cy} r="0.9"
        fill="rgba(95,70,148,0.18)" />;
      })
      )}

      {/* dispatcher → fan bus connector */}
      <line x1={G.node.x + 30} y1={G.node.y} x2={G.fanBusX} y2={G.node.y}
      stroke={accent} strokeWidth="1.4" opacity="0.6" />
      {/* fan bus (vertical) */}
      <line x1={G.fanBusX} y1={pickerCenterY(0)} x2={G.fanBusX} y2={pickerCenterY(G.pickN - 1)}
      stroke={accent} strokeWidth="1.6" opacity="0.55" />
      {/* 10 fan stubs from bus → picker left edge */}
      {Array.from({ length: G.pickN }).map((_, i) =>
      <line key={i} x1={G.fanBusX} y1={pickerCenterY(i)}
      x2={G.pickInX} y2={pickerCenterY(i)}
      stroke={accent} strokeWidth="1.2" opacity="0.55" />
      )}

      {/* converge bus (vertical) + 10 input stubs */}
      <line x1={G.convBusX} y1={pickerCenterY(0)} x2={G.convBusX} y2={pickerCenterY(G.pickN - 1)}
      stroke="#1c1814" strokeWidth="1.5" opacity="0.45" />
      {Array.from({ length: G.pickN }).map((_, i) =>
      <line key={i} x1={G.pickOutX} y1={pickerCenterY(i)}
      x2={G.convBusX} y2={pickerCenterY(i)}
      stroke="#1c1814" strokeWidth="1" opacity="0.4" />
      )}

      {/* converge bus → 2 exit lines (shipped up to truck, breach down to pile) */}
      <g>
        {/* SHIPPED — extends all the way to the back of the loading truck */}
        <line x1={G.convBusX} y1={G.shipExitY} x2={G.shipDock.x - 36} y2={G.shipExitY}
        stroke="#2c7e58" strokeWidth="2.4" />
        <polygon points={`${G.shipDock.x - 36},${G.shipExitY - 5} ${G.shipDock.x - 28},${G.shipExitY} ${G.shipDock.x - 36},${G.shipExitY + 5}`} fill="#2c7e58" />
        <text x={G.convBusX + 10} y={G.shipExitY - 9}
        fontFamily="'IBM Plex Mono', monospace" fontSize="10" fontWeight="700"
        fill="#2c7e58" letterSpacing="1.5px">SHIPPED ↑  →  LOADING TRUCK</text>

        {/* BREACH — ends at the breach pile */}
        <line x1={G.convBusX} y1={G.breachExitY} x2={G.shipPile.x + 60} y2={G.breachExitY}
        stroke="#aa3b2c" strokeWidth="2.4" />
        <polygon points={`${G.shipPile.x + 60},${G.breachExitY - 5} ${G.shipPile.x + 68},${G.breachExitY} ${G.shipPile.x + 60},${G.breachExitY + 5}`} fill="#aa3b2c" />
        <text x={G.convBusX + 10} y={G.breachExitY + 15}
        fontFamily="'IBM Plex Mono', monospace" fontSize="10" fontWeight="700"
        fill="#aa3b2c" letterSpacing="1.5px">BREACH ↓</text>
      </g>

      {/* shipped pile frame + dock */}
      <rect x={G.shipPile.x - 8} y={G.shipPile.y - 12}
      width={G.shipPile.cols * G.shipPile.pitch + 16} height="208"
      fill="rgba(155,207,177,0.12)" stroke="#2c7e58" strokeWidth="0.7" strokeDasharray="4 3" rx="3" />
      <text x={G.shipPile.x} y={G.shipPile.y - 18}
      fontFamily="'IBM Plex Mono', monospace" fontSize="9.5" fontWeight="700"
      fill="#2c7e58" letterSpacing="1.5px">SHIPPED ON-TIME</text>
      {/* shipping outbound truck — also animated */}
      <g transform={`translate(${G.shipDock.x} ${G.shipDock.y})`}>
        <animateTransform attributeName="transform" type="translate"
        values={`${G.shipDock.x} ${G.shipDock.y}; ${G.shipDock.x} ${G.shipDock.y - 0.5}; ${G.shipDock.x} ${G.shipDock.y}`}
        dur="1.1s" repeatCount="indefinite" />
        <rect x="-34" y="-22" width="50" height="42" rx="2" fill="#fff" stroke="#1c1814" strokeWidth="1" />
        <line x1="-28" y1="-18" x2="-28" y2="18" stroke="#1c1814" strokeWidth="0.5" opacity="0.45" />
        <line x1="-14" y1="-18" x2="-14" y2="18" stroke="#1c1814" strokeWidth="0.5" opacity="0.45" />
        <line x1="0" y1="-18" x2="0" y2="18" stroke="#1c1814" strokeWidth="0.5" opacity="0.45" />
        <path d="M16 -4 L16 -20 L36 -20 L46 -6 L46 -4 Z" fill="#9fcfb1" stroke="#1c1814" strokeWidth="1" strokeLinejoin="round" />
        <rect x="20" y="-16" width="12" height="9" fill="#cfe6f1" stroke="#1c1814" strokeWidth="0.6" />
        <circle cx="46" cy="-12" r="1.6" fill="#fff5c5" stroke="#1c1814" strokeWidth="0.4" />
        <g transform="translate(-20 22)">
          <circle r="6" fill="#1c1814" />
          <circle r="2.5" fill="#5a5040" />
          <g><animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="1.0s" repeatCount="indefinite" />
            <line x1="-6" y1="0" x2="6" y2="0" stroke="#aab3c0" strokeWidth="0.7" />
            <line x1="0" y1="-6" x2="0" y2="6" stroke="#aab3c0" strokeWidth="0.7" />
          </g>
        </g>
        <g transform="translate(30 22)">
          <circle r="6" fill="#1c1814" />
          <circle r="2.5" fill="#5a5040" />
          <g><animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="1.0s" begin="-0.4s" repeatCount="indefinite" />
            <line x1="-6" y1="0" x2="6" y2="0" stroke="#aab3c0" strokeWidth="0.7" />
            <line x1="0" y1="-6" x2="0" y2="6" stroke="#aab3c0" strokeWidth="0.7" />
          </g>
        </g>
      </g>

      {/* breach pile frame */}
      <rect x={G.breachPile.x - 8} y={G.breachPile.y - 12}
      width={G.breachPile.cols * G.breachPile.pitch + 16} height="208"
      fill="rgba(227,168,150,0.12)" stroke="#aa3b2c" strokeWidth="0.7" strokeDasharray="4 3" rx="3" />
      <text x={G.breachPile.x} y={G.breachPile.y - 18}
      fontFamily="'IBM Plex Mono', monospace" fontSize="9.5" fontWeight="700"
      fill="#aa3b2c" letterSpacing="1.5px">SLA BREACH / SPOILED</text>

      {/* DAHS feedback loop — the hero feature */}
      {isDahs && <FeedbackLoop />}
    </g>);

}

/* dispatcher circle with the rule + prob bars */
function DispatcherNode({ interval, isDahs, accent }) {
  const rule = interval && interval.rule || "FIFO";
  const probs = interval && interval.probs;
  const chosen = window.DAHS_SIM.HEUR.indexOf(rule);
  return (
    <g>
      <circle cx={G.node.x} cy={G.node.y} r="32" fill="#fff"
      stroke={accent} strokeWidth="1.8" />
      <circle cx={G.node.x} cy={G.node.y} r="27"
      fill={isDahs ? "rgba(33,104,69,0.13)" : "rgba(160,91,31,0.1)"} stroke="none" />
      <text x={G.node.x} y={G.node.y - 7} textAnchor="middle"
      fontFamily="'IBM Plex Mono', monospace" fontSize="8" fontWeight="600"
      fill={accent} letterSpacing="0.5px">
        {isDahs ? "DAHS" : "RULE"}
      </text>
      <text x={G.node.x} y={G.node.y + 9} textAnchor="middle"
      fontFamily="'IBM Plex Mono', monospace" fontSize="15" fontWeight="700"
      fill="#1c1814">{rule}</text>
      {probs &&
      <g transform={`translate(${G.node.x - 54} ${G.node.y + 40})`}>
          <rect x="-6" y="-2" width="124" height="48" rx="3"
        fill="rgba(255,253,245,0.95)" stroke={accent}
        strokeWidth="0.8" strokeOpacity="0.55" />
          <text x="2" y="8" fontFamily="'IBM Plex Mono', monospace"
        fontSize="7" fontWeight="600" fill="#1c1814" letterSpacing="0.5px">
            P(rule | state)
          </text>
          {window.DAHS_SIM.HEUR.map((h, i) => {
          const p = probs[i] || 0;
          const on = i === chosen;
          return (
            <g key={h} transform={`translate(${i * 29} 12)`}>
                <rect x="0" y="0" width="23" height="24" fill="rgba(28,24,20,0.05)" />
                <rect x="0" y={24 - p * 24} width="23" height={p * 24}
              fill={on ? accent : "rgba(28,24,20,0.3)"} />
                <text x="11" y="34" textAnchor="middle"
              fontFamily="'IBM Plex Mono', monospace" fontSize="7"
              fontWeight={on ? "700" : "500"}
              fill={on ? accent : "rgba(28,24,20,0.5)"}>{h}</text>
              </g>);

        })}
        </g>
      }
    </g>);

}

/* 10 picker boxes — vertical stack */
function PickerStack({ busy, accent }) {
  return (
    <g>
      {Array.from({ length: G.pickN }).map((_, i) => {
        const y = G.pickY0 + i * G.pickRowH;
        const on = busy[i];
        return (
          <g key={i}>
            <rect x={G.pickBoxX} y={y + 4} width={G.pickBoxW} height={G.pickBoxH}
            rx="3"
            fill={on ? "rgba(232,194,131,0.22)" : "rgba(255,253,245,0.6)"}
            stroke={on ? accent : "rgba(28,24,20,0.18)"}
            strokeWidth={on ? 1.2 : 0.8} />
            <text x={G.pickBoxX + 10} y={y + G.pickBoxH / 2 + 8}
            fontFamily="'IBM Plex Mono', monospace" fontSize="11"
            fontWeight="700"
            fill={on ? accent : "rgba(28,24,20,0.55)"}>
              {`P${String(i + 1).padStart(2, "0")}`}
            </text>
            <g transform={`translate(${G.pickBoxX + G.pickBoxW - 22} ${y + G.pickBoxH / 2 + 4})`}>
              <circle cx="0" cy="-10" r="4.5"
              fill={on ? "#d6b582" : "#cfcec7"}
              stroke="#1c1814" strokeWidth="0.7" />
              <path d="M -7 13 L -6 -2 Q 0 -5 6 -2 L 7 13 Z"
              fill={on ? accent : "#a4a298"}
              stroke="#1c1814" strokeWidth="0.7" strokeLinejoin="round" />
            </g>
          </g>);

      })}
    </g>);

}

/* the floor as a whole */
function FloorFigure({ run, t, role, layoutMode }) {
  const S = window.DAHS_SIM;
  const isDahs = role === "dahs";
  const accent = isDahs ? "#216845" : "#a05b1f";

  const iv = S.clamp(Math.floor(t / S.IV_MIN), 0, S.N_IV - 1);
  const interval = run.intervals[iv];

  const { vizList, busy } = React.useMemo(() => {
    const slotMap = S.buildSlotMap(run.orders, t);
    const list = [];const bz = new Array(G.pickN).fill(false);
    for (const o of run.orders) {
      const vz = orderViz(o, t, slotMap);
      if (!vz) continue;
      list.push({ o, vz });
      if (vz.phase === "processing" || vz.phase === "dispatch") bz[o.picker] = true;
    }
    const rank = { queued: 0, arriving: 1, drop: 1, breachPile: 1, docked: 1,
      processing: 2, outbound: 3, dispatch: 4 };
    list.sort((a, b) => (rank[a.vz.phase] || 0) - (rank[b.vz.phase] || 0));
    return { vizList: list, busy: bz };
  }, [run, t]);

  return (
    <svg viewBox={`0 0 ${G.W} ${G.H}`} preserveAspectRatio="xMidYMid meet">
      <FloorChrome role={role} />
      <DispatcherNode interval={interval} isDahs={isDahs} accent={accent} />
      <PickerStack busy={busy} accent={accent} />
      {vizList.map(({ o, vz }) => <Pkg key={o.id} vz={vz} o={o} />)}
    </svg>);

}

window.DAHS_FLOOR = { FloorFigure, G };