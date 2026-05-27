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
/* Package product categories, derived from each order's real attributes
 * (perishability + priority). The body colour tracks the category so a given
 * package TYPE is followable from arrival all the way to its outbound pile.
 * Outcome (shipped / breach / spoiled) stays legible through the destination
 * pile and an edge override below — it is no longer carried by the body fill. */
const CATEGORY = {
  perishable: { label: "Perishable · cold-chain", body: "#2bb3a3", edge: "#15756a" },
  express:    { label: "Express · high priority", body: "#9d5ce0", edge: "#5e2f9e" },
  standard:   { label: "Standard",                body: "#4f93e0", edge: "#2a5e9e" },
  economy:    { label: "Economy · low priority",  body: "#9aa6b6", edge: "#5f6b7a" },
};
const CAT_ORDER = ["perishable", "express", "standard", "economy"];
function categoryOf(o) {
  if (!o) return "standard";
  if (o.perishable) return "perishable";          // cold-chain dominates the type
  if (o.priority === "high") return "express";
  if (o.priority === "low") return "economy";
  return "standard";
}

const PRIO = { low: "#8b97ab", medium: "#c89238", high: "#aa3b2c" };

function Pkg({ vz, o }) {
  const cat = CATEGORY[categoryOf(o)] || CATEGORY.standard;
  // outcome state overrides the edge so shipped/breach/spoiled stay readable
  const breach = vz.tone === "breach";
  const spoiled = vz.tone === "spoiled";
  const edge = breach ? "#aa3b2c" : spoiled ? "#5e572f" : cat.edge;
  const r = 13;
  const C = 2 * Math.PI * 9;
  return (
    <g transform={`translate(${vz.x.toFixed(2)} ${vz.y.toFixed(2)})`}>
      <rect x={-r / 2} y={-r / 2} width={r} height={r}
      fill={cat.body} stroke={edge}
      strokeWidth={breach || spoiled ? 1.5 : 0.9}
      strokeDasharray={spoiled ? "2 1.5" : undefined} />
      <line x1="0" y1={-r / 2} x2="0" y2={r / 2} stroke="#fff8e3" strokeWidth="0.6" opacity="0.85" />
      <line x1={-r / 2} y1="0" x2={r / 2} y2="0" stroke="#fff8e3" strokeWidth="0.6" opacity="0.85" />
      {o &&
      <circle cx={r / 2 - 2.5} cy={-r / 2 + 2.5} r="2"
      fill={PRIO[o.priority] || PRIO.medium}
      stroke="#fff" strokeWidth="0.5" />
      }
      {vz.phase === "processing" &&
      <circle r="9" fill="none" stroke="#a06b27" strokeWidth="2"
      strokeLinecap="round" strokeDasharray={`${C}`}
      strokeDashoffset={`${C * (1 - (vz.prog || 0))}`}
      transform="rotate(-90)" />
      }
    </g>);

}

/* ────────── reusable wheel ────────── */
function TruckWheel({ cx, cy, r, delay = "0s", dur = "1.1s" }) {
  const R = +r;
  return (
    <g transform={`translate(${+cx} ${+cy})`}>
      <circle r={R + 0.6} fill="#1c1814" />
      <circle r={R} fill="#2a2522" stroke="#1c1814" strokeWidth="0.4" />
      <circle r={R * 0.48} fill="#7a6f5a" stroke="#1c1814" strokeWidth="0.4" />
      <circle r={R * 0.2} fill="#1c1814" />
      <g>
        <animateTransform attributeName="transform" type="rotate"
          from="0" to="360" dur={dur} begin={delay} repeatCount="indefinite" />
        <line x1={-R * 0.78} y1="0" x2={R * 0.78} y2="0" stroke="#c2cbd4" strokeWidth="0.65" />
        <line x1="0" y1={-R * 0.78} x2="0" y2={R * 0.78} stroke="#c2cbd4" strokeWidth="0.65" />
        <line x1={-R * 0.55} y1={-R * 0.55} x2={R * 0.55} y2={R * 0.55} stroke="#9aa3ad" strokeWidth="0.45" />
        <line x1={-R * 0.55} y1={R * 0.55} x2={R * 0.55} y2={-R * 0.55} stroke="#9aa3ad" strokeWidth="0.45" />
      </g>
    </g>
  );
}

/* ────────── animated inbound truck ──────────
 * Side-view semi-trailer. Cab on the right (already left the dock), trailer
 * back doors on the left feeding the conveyor. */
function InboundTruck({ x, y }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      {/* dock pad */}
      <rect x="-82" y="-108" width="164" height="216" rx="3"
        fill="#ecd9b1" stroke="#9a7a3a" strokeWidth="0.8" />
      {Array.from({ length: 16 }).map((_, i) =>
        <line key={i} x1="-76" y1={-100 + i * 13} x2="76" y2={-100 + i * 13}
          stroke="#bba474" strokeWidth="1.1" opacity="0.7" />
      )}
      {/* caution stripes */}
      <rect x="-82" y="-108" width="164" height="3" fill="#d6a93b" />
      <rect x="-82" y="105" width="164" height="3" fill="#d6a93b" />
      <text x="0" y="-116" textAnchor="middle"
        fontFamily="'IBM Plex Mono', monospace" fontSize="9"
        fill="rgba(28,24,20,0.6)" letterSpacing="2.5px" fontWeight="600">
        RECEIVING DOCK
      </text>

      {/* drop shadow under truck */}
      <ellipse cx="-12" cy="33" rx="62" ry="3.5" fill="rgba(28,24,20,0.18)" />

      {/* truck body (gentle idle bounce) */}
      <g>
        <animateTransform attributeName="transform" type="translate"
          values="0 0; 0 -0.5; 0 0" dur="1.1s" repeatCount="indefinite" />

        {/* TRAILER */}
        <rect x="-66" y="-24" width="68" height="48" rx="1.5"
          fill="#f6ecca" stroke="#1c1814" strokeWidth="1.1" />
        {/* roof highlight + bottom shadow */}
        <rect x="-66" y="-24" width="68" height="4" fill="#fdf6df" />
        <rect x="-66" y="20" width="68" height="4" fill="#cdbf99" opacity="0.7" />
        {/* roll-up cargo door (left = trailer back) */}
        <rect x="-66" y="-22" width="14" height="44" fill="#ebdfb8"
          stroke="#1c1814" strokeWidth="0.7" />
        {[0,1,2,3,4].map(i =>
          <line key={i} x1="-65" y1={-18 + i * 9} x2="-53" y2={-18 + i * 9}
            stroke="#1c1814" strokeWidth="0.4" opacity="0.55" />
        )}
        <circle cx="-55" cy="0" r="1.4" fill="#1c1814" />
        {/* side ribs */}
        {[-44, -34, -24, -14, -4].map(xx =>
          <line key={xx} x1={xx} y1="-19" x2={xx} y2="19"
            stroke="#1c1814" strokeWidth="0.3" opacity="0.3" />
        )}
        {/* reflectors */}
        <rect x="-62" y="-23" width="2.5" height="2.5" fill="#aa3b2c" />
        <rect x="-3" y="-23" width="2.5" height="2.5" fill="#aa3b2c" />
        <rect x="-62" y="20.5" width="2.5" height="2.5" fill="#d6a93b" />
        <rect x="-3" y="20.5" width="2.5" height="2.5" fill="#d6a93b" />
        {/* livery panel */}
        <rect x="-38" y="-8" width="34" height="16" fill="#fbf7e8"
          stroke="#9a7a3a" strokeWidth="0.45" />
        <text x="-21" y="4" textAnchor="middle"
          fontFamily="'Playfair Display', serif" fontSize="10"
          fontWeight="700" fill="#1c1814" letterSpacing="2px">DAHS</text>

        {/* TRACTOR CAB */}
        {/* roof strip */}
        <rect x="2" y="-26" width="28" height="3" rx="0.5" fill="#1c1814" />
        {/* cab body */}
        <path d="M 2 -23 L 2 18 L 30 18 L 30 -23 Z"
          fill="#e6d6a8" stroke="#1c1814" strokeWidth="1" />
        {/* hood */}
        <path d="M 30 -8 L 30 18 L 47 18 L 49 4 L 47 -4 L 36 -8 Z"
          fill="#d6c191" stroke="#1c1814" strokeWidth="1" strokeLinejoin="round" />
        {/* windshield */}
        <path d="M 6 -19 L 28 -19 L 28 -3 L 6 -3 Z"
          fill="#cfe6f1" stroke="#1c1814" strokeWidth="0.6" />
        <line x1="17" y1="-19" x2="17" y2="-3" stroke="#1c1814" strokeWidth="0.4" />
        {/* side mirror (sticks out from cab) */}
        <rect x="0.5" y="-15" width="1.5" height="9" fill="#1c1814" />
        <rect x="-3" y="-14" width="3.5" height="6" fill="#cfe6f1"
          stroke="#1c1814" strokeWidth="0.4" />
        {/* door outline + handle */}
        <rect x="6" y="-1" width="20" height="18" fill="none"
          stroke="#1c1814" strokeWidth="0.45" opacity="0.55" />
        <rect x="21" y="6" width="3" height="1.2" fill="#1c1814" />
        {/* grille */}
        <rect x="40" y="-1" width="7" height="11" fill="#1c1814" />
        {[1,2,3,4,5].map(i =>
          <line key={i} x1="40" y1={-1 + i * 2} x2="47" y2={-1 + i * 2}
            stroke="#7a6a52" strokeWidth="0.3" />
        )}
        {/* headlight */}
        <ellipse cx="47" cy="-3" rx="2.4" ry="2" fill="#fff5c5"
          stroke="#1c1814" strokeWidth="0.5" />
        {/* bumper + plate */}
        <rect x="44" y="13" width="6" height="5" fill="#3a342a"
          stroke="#1c1814" strokeWidth="0.4" />
        <rect x="42" y="9" width="6" height="3" fill="#fbf7e8"
          stroke="#1c1814" strokeWidth="0.3" />
      </g>

      {/* exhaust + smoke */}
      <rect x="29" y="-32" width="2.5" height="9" fill="#5a5040"
        stroke="#1c1814" strokeWidth="0.4" />
      <rect x="28.5" y="-33" width="3.5" height="1.5" fill="#1c1814" />
      {[0, 1, 2].map((i) =>
        <circle key={i} cx="30.5" cy="-33" r="2.5" fill="#b9b3a4" opacity="0">
          <animate attributeName="cy" values="-33;-58" dur="1.8s"
            begin={`${i * 0.6}s`} repeatCount="indefinite" />
          <animate attributeName="cx" values="30.5;24" dur="1.8s"
            begin={`${i * 0.6}s`} repeatCount="indefinite" />
          <animate attributeName="r" values="2;6.5" dur="1.8s"
            begin={`${i * 0.6}s`} repeatCount="indefinite" />
          <animate attributeName="opacity" values="0;0.5;0" dur="1.8s"
            begin={`${i * 0.6}s`} repeatCount="indefinite" />
        </circle>
      )}

      {/* WHEELS — 2 trailer axles, cab drive axle, cab steer axle */}
      <TruckWheel cx="-32" cy="22" r="7" delay="-0.6s" />
      <TruckWheel cx="-14" cy="22" r="7" delay="-0.4s" />
      <TruckWheel cx="14"  cy="22" r="7" delay="-0.2s" />
      <TruckWheel cx="38"  cy="22" r="6.5" />
      {/* mud flap behind rear axle */}
      <rect x="-43" y="22" width="2" height="11" fill="#1c1814" />

      {/* "next in line" silhouette */}
      <g transform="translate(-2 -74)" opacity="0.32">
        <rect x="-46" y="-12" width="52" height="24" fill="#c7baa0"
          stroke="#1c1814" strokeWidth="0.6" />
        <path d="M 6 -8 L 6 12 L 26 12 L 32 4 L 32 -4 L 22 -8 Z"
          fill="#bfb29a" stroke="#1c1814" strokeWidth="0.6"
          strokeLinejoin="round" />
        <circle cx="-30" cy="14" r="3.5" fill="#1c1814" />
        <circle cx="16" cy="14" r="3.5" fill="#1c1814" />
        <text x="-12" y="-18" textAnchor="middle"
          fontFamily="'IBM Plex Mono', monospace" fontSize="6.5"
          fill="rgba(28,24,20,0.7)" letterSpacing="1.5px">NEXT IN LINE</text>
      </g>
    </g>
  );
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
      {/* shipping outbound truck — loaded with completed orders, cab on right */}
      <g transform={`translate(${G.shipDock.x} ${G.shipDock.y})`}>
        {/* drop shadow */}
        <ellipse cx="-2" cy="33" rx="52" ry="3" fill="rgba(28,24,20,0.18)" />

        <g>
          <animateTransform attributeName="transform" type="translate"
            values="0 0; 0 -0.4; 0 0" dur="1.1s" repeatCount="indefinite" />

          {/* TRAILER */}
          <rect x="-50" y="-22" width="62" height="44" rx="1.5"
            fill="#e6f1ea" stroke="#1c1814" strokeWidth="1" />
          <rect x="-50" y="-22" width="62" height="4" fill="#f4faf6" />
          <rect x="-50" y="18" width="62" height="4" fill="#b3cebd" opacity="0.65" />
          {/* roll-up door (left = back of trailer, facing warehouse) */}
          <rect x="-50" y="-20" width="12" height="40" fill="#d6e6dc"
            stroke="#1c1814" strokeWidth="0.6" />
          {[0,1,2,3,4].map(i =>
            <line key={i} x1="-49" y1={-16 + i * 8} x2="-38" y2={-16 + i * 8}
              stroke="#1c1814" strokeWidth="0.4" opacity="0.55" />
          )}
          <circle cx="-40" cy="0" r="1.2" fill="#1c1814" />
          {/* side ribs */}
          {[-30, -22, -14, -6, 2].map(xx =>
            <line key={xx} x1={xx} y1="-17" x2={xx} y2="17"
              stroke="#1c1814" strokeWidth="0.3" opacity="0.3" />
          )}
          {/* reflectors */}
          <rect x="-46" y="-21" width="2" height="2.5" fill="#aa3b2c" />
          <rect x="8" y="-21" width="2" height="2.5" fill="#aa3b2c" />
          <rect x="-46" y="18.5" width="2" height="2.5" fill="#d6a93b" />
          <rect x="8" y="18.5" width="2" height="2.5" fill="#d6a93b" />
          {/* livery */}
          <rect x="-28" y="-7" width="34" height="14" fill="#fbf7e8"
            stroke="#2c7e58" strokeWidth="0.45" />
          <text x="-11" y="3.5" textAnchor="middle"
            fontFamily="'Playfair Display', serif" fontSize="9"
            fontWeight="700" fill="#2c7e58" letterSpacing="1.8px">SHIPPED</text>

          {/* TRACTOR CAB (right side) */}
          <rect x="12" y="-24" width="24" height="3" rx="0.5" fill="#1c1814" />
          <path d="M 12 -21 L 12 18 L 36 18 L 36 -21 Z"
            fill="#9fcfb1" stroke="#1c1814" strokeWidth="1" />
          {/* hood */}
          <path d="M 36 -7 L 36 18 L 51 18 L 53 4 L 51 -3 L 42 -7 Z"
            fill="#85bc9b" stroke="#1c1814" strokeWidth="1" strokeLinejoin="round" />
          {/* windshield */}
          <path d="M 15 -18 L 34 -18 L 34 -3 L 15 -3 Z"
            fill="#cfe6f1" stroke="#1c1814" strokeWidth="0.55" />
          <line x1="24.5" y1="-18" x2="24.5" y2="-3"
            stroke="#1c1814" strokeWidth="0.4" />
          {/* mirror */}
          <rect x="10.5" y="-14" width="1.5" height="8" fill="#1c1814" />
          <rect x="7" y="-13" width="3.5" height="5" fill="#cfe6f1"
            stroke="#1c1814" strokeWidth="0.4" />
          {/* door */}
          <rect x="16" y="-1" width="17" height="17" fill="none"
            stroke="#1c1814" strokeWidth="0.45" opacity="0.55" />
          <rect x="28" y="6" width="3" height="1.2" fill="#1c1814" />
          {/* grille */}
          <rect x="44" y="-1" width="7" height="11" fill="#1c1814" />
          {[1,2,3,4,5].map(i =>
            <line key={i} x1="44" y1={-1 + i * 2} x2="51" y2={-1 + i * 2}
              stroke="#7a6a52" strokeWidth="0.3" />
          )}
          {/* headlight */}
          <ellipse cx="51" cy="-3" rx="2.2" ry="1.9" fill="#fff5c5"
            stroke="#1c1814" strokeWidth="0.5" />
          {/* bumper + plate */}
          <rect x="48" y="13" width="5" height="5" fill="#3a342a"
            stroke="#1c1814" strokeWidth="0.4" />
          <rect x="46" y="9" width="6" height="3" fill="#fbf7e8"
            stroke="#1c1814" strokeWidth="0.3" />
        </g>

        {/* exhaust + smoke */}
        <rect x="35" y="-31" width="2.2" height="8" fill="#5a5040"
          stroke="#1c1814" strokeWidth="0.4" />
        <rect x="34.6" y="-32" width="3" height="1.3" fill="#1c1814" />
        {[0, 1, 2].map((i) =>
          <circle key={i} cx="36" cy="-32" r="2" fill="#b9b3a4" opacity="0">
            <animate attributeName="cy" values="-32;-54" dur="1.8s"
              begin={`${i * 0.6 + 0.3}s`} repeatCount="indefinite" />
            <animate attributeName="cx" values="36;30" dur="1.8s"
              begin={`${i * 0.6 + 0.3}s`} repeatCount="indefinite" />
            <animate attributeName="r" values="1.8;5.5" dur="1.8s"
              begin={`${i * 0.6 + 0.3}s`} repeatCount="indefinite" />
            <animate attributeName="opacity" values="0;0.45;0" dur="1.8s"
              begin={`${i * 0.6 + 0.3}s`} repeatCount="indefinite" />
          </circle>
        )}

        {/* WHEELS */}
        <TruckWheel cx="-30" cy="22" r="6.5" delay="-0.6s" />
        <TruckWheel cx="-14" cy="22" r="6.5" delay="-0.4s" />
        <TruckWheel cx="22"  cy="22" r="6.5" delay="-0.2s" />
        <TruckWheel cx="44"  cy="22" r="6" />
        <rect x="-39" y="22" width="2" height="10" fill="#1c1814" />
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

window.DAHS_FLOOR = { FloorFigure, G, CATEGORY, CAT_ORDER, categoryOf };