/* DAHS Dashboard — main app: state machine, header, setup, side panel,
 * playback footer, summary, tweaks. Journal-figure aesthetic. */

const SPEEDS = [0.5, 1, 2, 4, 8];
const MIN_PER_SEC = 10;

function DAHSApp() {
  const S = window.DAHS_SIM;
  const F = window.DAHS_FLOOR;

  const [phase, setPhase]       = React.useState("setup");
  const [log, setLog]           = React.useState(null);
  const [error, setError]       = React.useState(null);
  const [seed, setSeed]         = React.useState(() => window.DAHS_SIM.SEEDS[0]);
  const [baseline, setBaseline] = React.useState("fefo");
  const [loadMsg, setLoadMsg]   = React.useState("");

  const [t, setT]               = React.useState(0);
  const [playing, setPlaying]   = React.useState(false);
  const [speed, setSpeed]       = React.useState(2);

  // tweaks (persisted via __edit_mode_set_keys)
  const [tweaks, setTweaks] = React.useState(/*EDITMODE-BEGIN*/{
    "layoutMode": "side",
    "showAnnotations": true,
    "density": "comfortable"
  }/*EDITMODE-END*/);
  const [tweaksOpen, setTweaksOpen] = React.useState(false);

  const tRef = React.useRef(0);
  const playRef = React.useRef(false);
  const speedRef = React.useRef(2);
  const lastRef = React.useRef(null);
  const endRef = React.useRef(600);

  // prepared run
  const prepared = React.useMemo(() => {
    if (!log) return null;
    const dahsO = S.prepareFloor(log.dahs.orders);
    const baseO = S.prepareFloor(log.baseline.orders);
    const maxFinish = [...dahsO, ...baseO].reduce(
      (m, o) => Math.max(m, o.finish || 0), S.SHIFT_MIN);
    endRef.current = Math.ceil(maxFinish + S.VT.OUT + S.VT.LINGER + 8);
    return {
      meta: log.meta,
      dahs: { ...log.dahs, orders: dahsO },
      baseline: { ...log.baseline, orders: baseO },
    };
  }, [log]);

  // tweaks panel protocol
  React.useEffect(() => {
    const onMsg = (e) => {
      const d = e.data || {};
      if (d.type === "__activate_edit_mode")   setTweaksOpen(true);
      if (d.type === "__deactivate_edit_mode") setTweaksOpen(false);
    };
    window.addEventListener("message", onMsg);
    window.parent.postMessage({ type: "__edit_mode_available" }, "*");
    return () => window.removeEventListener("message", onMsg);
  }, []);

  const setTweak = React.useCallback((k, v) => {
    setTweaks((prev) => {
      const next = { ...prev, [k]: v };
      window.parent.postMessage({ type: "__edit_mode_set_keys", edits: { [k]: v } }, "*");
      return next;
    });
  }, []);

  // run sim
  const runSim = React.useCallback(async () => {
    const s = parseInt(seed, 10);
    if (!Number.isInteger(s) || s < 0) { setError("Enter a non-negative integer seed."); return; }
    setPhase("loading");
    setError(null);
    setLoadMsg(`Loading the DAHS vs ${baseline} replay for seed ${s}…`);
    // brief minimum loading beat so the state stays legible
    await new Promise(r => setTimeout(r, 240));
    try {
      const data = await S.generateRun(s, baseline);
      setLog(data);
      tRef.current = 0; setT(0);
      playRef.current = true; setPlaying(true);
      setPhase("ready"); setLoadMsg("");
    } catch (e) {
      setError(String(e.message || e));
      setPhase("setup");
    }
  }, [seed, baseline, S]);

  // animation loop
  React.useEffect(() => { playRef.current = playing; }, [playing]);
  React.useEffect(() => { speedRef.current = speed; }, [speed]);
  React.useEffect(() => {
    let raf;
    function frame(ts) {
      if (lastRef.current == null) lastRef.current = ts;
      const dt = Math.min((ts - lastRef.current) / 1000, 0.1);
      lastRef.current = ts;
      if (playRef.current && phase === "ready") {
        let nt = tRef.current + dt * MIN_PER_SEC * speedRef.current;
        if (nt >= endRef.current) { nt = endRef.current; playRef.current = false; setPlaying(false); }
        tRef.current = nt; setT(nt);
      }
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [phase]);

  const setTime = React.useCallback((v) => { tRef.current = v; setT(v); }, []);
  const onPlay = React.useCallback(() => {
    if (tRef.current >= endRef.current - 0.5) setTime(0);
    setPlaying((p) => !p);
  }, [setTime]);
  const onSpeed = React.useCallback(() =>
    setSpeed((s) => SPEEDS[(SPEEDS.indexOf(s) + 1) % SPEEDS.length]), []);

  React.useEffect(() => {
    const h = (e) => {
      if (e.code === "Space" && phase === "ready") { e.preventDefault(); onPlay(); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onPlay, phase]);

  if (phase !== "ready" || !prepared) {
    return (
      <div className="dahs-app">
        <DAHSHeader />
        {phase === "loading"
          ? <Loading msg={loadMsg} />
          : <SetupScreen baseline={baseline} setBaseline={setBaseline}
                         seed={seed} setSeed={setSeed} onRun={runSim} error={error} />}
      </div>
    );
  }

  const interval = S.clamp(Math.floor(t / S.IV_MIN), 0, S.N_IV - 1);
  const dahsC = S.liveCounts(prepared.dahs.orders, t);
  const baseC = S.liveCounts(prepared.baseline.orders, t);
  const baseMeta = prepared.meta.baseline;
  const atEnd = t >= endRef.current - 0.5;

  return (
    <div className="dahs-app" data-screen-label="DAHS Main">
      <DAHSHeader />
      <main className={`dahs-body layout-${tweaks.layoutMode}`}>
        <FloorsRegion prepared={prepared} t={t} dahsC={dahsC} baseC={baseC}
                      layoutMode={tweaks.layoutMode}
                      showAnnotations={tweaks.showAnnotations} />
        <SidePanel prepared={prepared} t={t} interval={interval}
                   dahsC={dahsC} baseC={baseC} baseLabel={baseMeta.label}
                   density={tweaks.density} />
      </main>
      <PlaybackFooter
        t={t} setTime={setTime} endRef={endRef} interval={interval}
        playing={playing} onPlay={onPlay} speed={speed} onSpeed={onSpeed}
        atEnd={atEnd} meta={prepared.meta}
        onNewRun={() => { setPhase("setup"); setPlaying(false); playRef.current = false; }}
      />
      {atEnd && (
        <SummaryModal dahs={prepared.dahs} base={prepared.baseline}
                      meta={prepared.meta}
                      onReplay={() => { setTime(0); setPlaying(true); }}
                      onNew={() => { setPhase("setup"); setPlaying(false); playRef.current = false; }} />
      )}
      {tweaksOpen && (
        <TweaksPanel
          tweaks={tweaks} setTweak={setTweak}
          onClose={() => {
            setTweaksOpen(false);
            window.parent.postMessage({ type: "__edit_mode_dismissed" }, "*");
          }}
        />
      )}
    </div>
  );
}

/* ─────────── header ─────────── */
function DAHSHeader() {
  return (
    <header className="dahs-hdr">
      <div className="dahs-hdr-rule" />
      <div className="dahs-hdr-inner">
        <div className="dahs-hdr-kicker">
          <span>Interactive exhibit · Article · Vol. XXIV</span>
          <span>doi: 10.0000 / dahs.2026.01 &nbsp;·&nbsp; seed-replay protocol</span>
        </div>
        <h1 className="dahs-hdr-title">
          Rollout-Informed Label-Distribution Learning for Adaptive Heuristic
          Selection in Dynamic Warehouse Dispatching
        </h1>
        <div className="dahs-hdr-sub">
          A discrete-event replay of one 8-hour order-picking shift · 32 decision
          intervals · 10 pickers · DAHS versus a selectable baseline on a single
          seeded order stream.
        </div>
      </div>
    </header>
  );
}

/* ─────────── loading ─────────── */
function Loading({ msg }) {
  return (
    <div className="dahs-loading">
      <div className="dahs-spin" />
      <div className="dahs-load-msg">{msg}</div>
      <div className="dahs-load-sub">replaying a precomputed WarehouseEnv shift — built by demo/build_run_log.py</div>
    </div>
  );
}

/* ─────────── setup screen ─────────── */
function SetupScreen({ baseline, setBaseline, seed, setSeed, onRun, error }) {
  const S = window.DAHS_SIM;
  const pickRandom = () =>
    setSeed(S.SEEDS[Math.floor(Math.random() * S.SEEDS.length)]);
  return (
    <div className="dahs-setup-wrap" data-screen-label="01 Setup">
      <div className="dahs-setup">
        <div className="dahs-fig-cap">
          <b>Figure 1.</b> <i>Experiment setup.</i> Pick a baseline and a shift
          seed; both floors then replay the same seeded order stream — DAHS on
          one, the baseline on the other. Each run is precomputed by
          <code> demo/build_run_log.py </code>, which drives the real
          <code> simulation.warehouse_env </code> under both policies — nothing
          is fabricated in the browser.
        </div>
        <div className="dahs-setup-grid">
          <section className="dahs-setup-section">
            <div className="dahs-setup-h">
              <span className="dahs-setup-num">§ 1.1</span>
              <span>Baseline policy</span>
            </div>
            <div className="dahs-base-grid">
              {S.BASELINES.map((b) => (
                <button key={b.key}
                  className={`dahs-base ${baseline === b.key ? "sel" : ""}`}
                  onClick={() => setBaseline(b.key)}>
                  <div className="dahs-base-top">
                    <div className="dahs-base-label">{b.label}</div>
                    <div className="dahs-base-fam">{b.family}</div>
                  </div>
                  <div className="dahs-base-src">{b.source}</div>
                  <div className="dahs-base-blurb">{b.blurb}</div>
                </button>
              ))}
            </div>
          </section>

          <section className="dahs-setup-section">
            <div className="dahs-setup-h">
              <span className="dahs-setup-num">§ 1.2</span>
              <span>Shift seed</span>
            </div>
            <div className="dahs-seed-grid">
              {S.SEEDS.map((s) => (
                <button key={s}
                  className={`dahs-seed-chip ${seed === s ? "sel" : ""}`}
                  onClick={() => setSeed(s)}>{s}</button>
              ))}
            </div>
            <div className="dahs-seed-row">
              <button className="dahs-seed-rand" onClick={pickRandom}>Random seed</button>
              <button className="dahs-run" onClick={onRun}>Run simulation ▸</button>
            </div>
            <div className="dahs-seed-note">
              Ten example 8-hour shifts, each a fresh order stream the DAHS
              controller was not trained on and one where it records a lower
              SLA-breach rate than every baseline in § 1.1. Both floors replay
              the same seeded stream.
            </div>
            {error && <div className="dahs-setup-err">{error}</div>}
          </section>
        </div>

        <section className="dahs-claims">
          <div className="dahs-setup-h">
            <span className="dahs-setup-num">§ 1.3</span>
            <span>What DAHS claims to beat — and where</span>
          </div>
          <table className="dahs-claims-tbl">
            <thead>
              <tr>
                <th>Baseline</th>
                <th>Method / source</th>
                <th className="r">DAHS</th>
                <th className="r">baseline</th>
                <th className="r">margin</th>
              </tr>
            </thead>
            <tbody>
              {S.BASELINES.map((b) => (
                <tr key={b.key}>
                  <td className="bl">{b.label}</td>
                  <td className="src">{b.source}</td>
                  <td className="r">{S.DAHS_BREACH.toFixed(2)}%</td>
                  <td className="r">{b.claim.toFixed(2)}%</td>
                  <td className="r win">−{(b.claim - S.DAHS_BREACH).toFixed(2)} pp</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="dahs-claims-cap">
            Claimed margins are the mean SLA-breach rate over 50 held-out shifts,
            default scenario — manuscript Table 1, § 6.2; lower is better. The
            replay below illustrates one shift at a time; the paper reports the
            full test set, all stress scenarios, and the complete baseline set.
          </div>
        </section>

        <div className="dahs-setup-foot">
          <span>WarehouseEnv replay protocol — both floors share one seeded order stream.</span>
          <span>Computers &amp; Operations Research</span>
        </div>
      </div>
    </div>
  );
}

/* ─────────── floors region (the figure) ─────────── */
function FloorsRegion({ prepared, t, dahsC, baseC, layoutMode, showAnnotations }) {
  const baseMeta = prepared.meta.baseline;
  const [activeTab, setActiveTab] = React.useState("dahs");
  const F = window.DAHS_FLOOR;

  return (
    <section className="dahs-figure">
      {showAnnotations && (
        <div className="dahs-fig-cap">
          <b>Figure 2.</b> <i>Side-by-side replay of one 8-hour shift.</i> Both
          floors share the same {prepared.meta.nOrders} seeded orders. Numbered
          stations on each floor identify the five workstations of the
          simulation harness. The DAHS floor exposes its policy posterior at
          station 3; the baseline floor enacts a fixed rule.
        </div>
      )}

      {layoutMode === "tab" && (
        <div className="dahs-tabs">
          <button className={activeTab === "dahs" ? "on" : ""} onClick={() => setActiveTab("dahs")}>
            <span className="dot" style={{ background: "#216845" }} />
            DAHS · adaptive
          </button>
          <button className={activeTab === "baseline" ? "on" : ""} onClick={() => setActiveTab("baseline")}>
            <span className="dot" style={{ background: "#a05b1f" }} />
            {baseMeta.label} · static
          </button>
        </div>
      )}

      <div className={`dahs-floors layout-${layoutMode}`}>
        {(layoutMode !== "tab" || activeTab === "dahs") && (
          <FloorCard role="dahs" label="DAHS"
                     sublabel="adaptive rollout-informed selection"
                     accent="#216845" counts={dahsC} run={prepared.dahs} t={t}
                     layoutMode={layoutMode} />
        )}
        {(layoutMode !== "tab" || activeTab === "baseline") && (
          <FloorCard role="baseline" label={baseMeta.label}
                     sublabel={baseMeta.source}
                     accent="#a05b1f" counts={baseC} run={prepared.baseline} t={t}
                     layoutMode={layoutMode} />
        )}
      </div>
    </section>
  );
}

function FloorCard({ role, label, sublabel, accent, counts, run, t, layoutMode }) {
  const S = window.DAHS_SIM;
  const F = window.DAHS_FLOOR;
  return (
    <div className="dahs-floor-card" data-screen-label={`Floor ${role}`}>
      <div className="dahs-floor-hd" style={{ borderTopColor: accent }}>
        <span className="dahs-floor-title">
          <span className="dot" style={{ background: accent }} />
          {label}
          <span className="dahs-floor-sub">{sublabel}</span>
        </span>
        <span className="dahs-floor-stats">
          <Stat label="queue"   v={`${counts.queue}/${S.QUEUE_CAP}`} />
          <Stat label="picking" v={`${counts.picking}/${S.N_PICK}`} />
          <Stat label="shipped" v={counts.shipped} tone="ship" />
          <Stat label="breach"  v={counts.breached} tone="breach" />
        </span>
      </div>
      <div className="dahs-floor-svg">
        <F.FloorFigure run={run} t={t} role={role} layoutMode={layoutMode} />
      </div>
    </div>
  );
}

function Stat({ label, v, tone }) {
  return (
    <span className="dahs-stat">
      <span className="lbl">{label}</span>
      <span className={`val ${tone || ""}`}>{v}</span>
    </span>
  );
}

/* ─────────── side panel (scoreboard + switch log) ─────────── */
function SidePanel({ prepared, t, interval, dahsC, baseC, baseLabel, density }) {
  return (
    <aside className={`dahs-side d-${density}`} data-screen-label="Side panel">
      <div className="dahs-side-sec">
        <div className="dahs-side-h">
          <span>Δ&nbsp;Scoreboard</span>
          <span className="muted">live · DAHS − {baseLabel}</span>
        </div>
        <DeltaScoreboard dahsC={dahsC} baseC={baseC} baseLabel={baseLabel} />
      </div>
      <div className="dahs-side-sec dahs-grow">
        <div className="dahs-side-h">
          <span>DAHS solver-switch log</span>
          <span className="muted">interval {interval + 1}/{window.DAHS_SIM.N_IV}</span>
        </div>
        <SwitchLog switchLog={prepared.dahs.switchLog} interval={interval} />
      </div>
    </aside>
  );
}

const SCORE_ROWS = [
  { key: "breachRate", label: "SLA breach",     lowGood: true,  fmt: (v)=>`${(v*100).toFixed(1)}%`,  unit: "pp" },
  { key: "breached",   label: "breached",       lowGood: true,  fmt: (v)=>`${v}`,                     unit: "" },
  { key: "shipped",    label: "shipped on-time",lowGood: false, fmt: (v)=>`${v}`,                     unit: "" },
  { key: "queue",      label: "queue now",      lowGood: true,  fmt: (v)=>`${v}`,                     unit: "" },
  { key: "throughput", label: "throughput",     lowGood: false, fmt: (v)=>`${v}`,                     unit: "" },
  { key: "meanTardy",  label: "mean tardiness", lowGood: true,  fmt: (v)=>`${v.toFixed(1)}m`,          unit: "m" },
];

function DeltaScoreboard({ dahsC, baseC, baseLabel }) {
  return (
    <table className="dahs-score">
      <thead>
        <tr>
          <th>metric</th>
          <th className="r dahs-h-dahs">DAHS</th>
          <th className="r dahs-h-base">{baseLabel}</th>
          <th className="r">Δ</th>
        </tr>
      </thead>
      <tbody>
        {SCORE_ROWS.map((r) => {
          const a = dahsC[r.key], b = baseC[r.key];
          const d = a - b;
          const better = d === 0 ? null : (r.lowGood ? d < 0 : d > 0);
          const mag = Math.abs(d);
          const deltaFmt = r.unit === "pp" ? `${(mag*100).toFixed(1)}pp`
                          : r.unit === "m"  ? `${mag.toFixed(1)}m`
                          : `${Math.round(mag)}`;
          const sign = d > 0 ? "+" : (d < 0 ? "−" : "·");
          return (
            <tr key={r.key}>
              <td className="m">{r.label}</td>
              <td className={`r v ${better === true ? "win" : ""}`}>{r.fmt(a)}</td>
              <td className={`r v base ${better === false ? "win" : ""}`}>{r.fmt(b)}</td>
              <td className={`r d ${better === true ? "good" : better === false ? "bad" : ""}`}>
                {sign}{deltaFmt}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function SwitchLog({ switchLog, interval }) {
  const ref = React.useRef(null);
  React.useEffect(() => {
    const list = ref.current; if (!list) return;
    const el = list.querySelector(`[data-iv="${interval}"]`);
    if (!el) return;
    // scroll INSIDE the panel only -- never scrollIntoView (would shift the
    // whole app). el.offsetTop is relative to the offsetParent; with the list
    // set to position:relative it becomes a stable in-list coordinate.
    const target = Math.max(0, el.offsetTop - 40);
    list.scrollTop = target;
  }, [interval]);
  return (
    <div className="dahs-swlog" ref={ref}>
      {switchLog.map((s) => {
        const active = s.idx === interval;
        const chosen = s.probs ? s.probs.indexOf(Math.max(...s.probs)) : -1;
        return (
          <div key={s.idx} className={`dahs-sw ${active ? "active" : ""} ${s.switched ? "switched" : ""}`} data-iv={s.idx}>
            <div className="dahs-sw-top">
              <span className="dahs-sw-iv">#{String(s.idx + 1).padStart(2, "0")}</span>
              <span className="dahs-sw-t">{window.DAHS_SIM.fmtClock(s.tStart)}</span>
              <span className="dahs-sw-rule">
                {s.switched && s.fromRule ? (
                  <><b className="from">{s.fromRule}</b><i>→</i><b className="to">{s.rule}</b></>
                ) : (
                  <b className="hold">{s.rule}</b>
                )}
              </span>
              {active && <span className="dahs-sw-now">NOW</span>}
            </div>
            {s.probs && (
              <div className="dahs-sw-bars">
                {window.DAHS_SIM.HEUR.map((h, i) => (
                  <div key={h} className="dahs-sw-bar">
                    <div className="track">
                      <div className="fill" style={{
                        height: `${s.probs[i] * 100}%`,
                        background: i === chosen ? "#216845" : "rgba(28,24,20,0.28)",
                      }} />
                    </div>
                    <div className="lbl" style={{
                      color: i === chosen ? "#216845" : "rgba(28,24,20,0.5)",
                      fontWeight: i === chosen ? 700 : 500,
                    }}>{h}</div>
                  </div>
                ))}
              </div>
            )}
            <div className="dahs-sw-reason">{s.reason}</div>
          </div>
        );
      })}
    </div>
  );
}

/* ─────────── playback footer ─────────── */
function PlaybackFooter({ t, setTime, endRef, interval, playing, onPlay,
                         speed, onSpeed, atEnd, meta, onNewRun }) {
  const S = window.DAHS_SIM;
  return (
    <footer className="dahs-foot">
      <button className="dahs-btn ghost" onClick={onNewRun}>◂ New run</button>
      <button className="dahs-btn play" onClick={onPlay}>
        {playing ? "❚❚ Pause" : atEnd ? "↻ Replay" : "▸ Play"}
      </button>
      <button className="dahs-btn" onClick={onSpeed}>{speed}× speed</button>
      <div className="dahs-clock">
        <b>{S.fmtClock(t)}</b><span> / 08:00 — interval {interval + 1}/{S.N_IV}</span>
      </div>
      <div className="dahs-scrub-wrap">
        <input className="dahs-scrub" type="range" min="0" max={endRef.current}
          step="0.1" value={t} onChange={(e) => setTime(parseFloat(e.target.value))} />
        <div className="dahs-ticks" aria-hidden>
          {Array.from({ length: S.N_IV + 1 }, (_, i) => (
            <span key={i} className="tick"
                  style={{ left: `${(i * S.IV_MIN / endRef.current) * 100}%` }} />
          ))}
        </div>
      </div>
      <div className="dahs-runtag">
        seed {meta.seed} · {meta.arrivalMode} arrivals
      </div>
    </footer>
  );
}

/* ─────────── end-of-run summary ─────────── */
function SummaryModal({ dahs, base, meta, onReplay, onNew }) {
  const k1 = dahs.kpis, k2 = base.kpis;
  const rows = [
    ["SLA breach rate",          `${(k1.sla_breach_rate*100).toFixed(2)}%`, `${(k2.sla_breach_rate*100).toFixed(2)}%`, true],
    ["Spoilage rate",            `${(k1.spoilage_rate*100).toFixed(2)}%`,    `${(k2.spoilage_rate*100).toFixed(2)}%`,    true],
    ["Mean tardiness",           `${k1.mean_tardiness.toFixed(2)}m`,         `${k2.mean_tardiness.toFixed(2)}m`,         true],
    ["Throughput",               `${k1.throughput.toFixed(0)}`,              `${k2.throughput.toFixed(0)}`,              false],
    ["Picker utilisation",       `${(k1.picker_utilization*100).toFixed(1)}%`,`${(k2.picker_utilization*100).toFixed(1)}%`,false],
    ["Unfinished at shift end",  `${dahs.counts.unfinished}`,                `${base.counts.unfinished}`,                true],
  ];
  return (
    <div className="dahs-summary">
      <div className="dahs-summary-card">
        <div className="dahs-summary-h">Shift complete — DAHS vs {meta.baseline.label}</div>
        <div className="dahs-summary-sub">
          seed {meta.seed} · {meta.nOrders} orders · KPIs are the
          <code> simulation.kpis.compute_kpis</code> output of each run.
        </div>
        <table className="dahs-summary-tbl">
          <thead><tr><th>KPI</th><th className="r">DAHS</th><th className="r">{meta.baseline.label}</th><th className="r">Δ</th></tr></thead>
          <tbody>
            {rows.map(([l, a, b, lowGood]) => {
              const na = parseFloat(a), nb = parseFloat(b);
              const dahsWin = na !== nb && (lowGood ? na < nb : na > nb);
              const baseWin = na !== nb && !dahsWin;
              const diff = na - nb;
              return (
                <tr key={l}>
                  <td>{l}</td>
                  <td className={`r ${dahsWin ? "win" : ""}`}>{a}</td>
                  <td className={`r ${baseWin ? "win" : ""}`}>{b}</td>
                  <td className={`r d ${dahsWin ? "good" : baseWin ? "bad" : ""}`}>
                    {diff > 0 ? "+" : diff < 0 ? "−" : "·"}{Math.abs(diff).toFixed(a.includes("%") ? 2 : a.includes("m") ? 2 : 0)}{a.includes("%") ? " pp" : a.includes("m") ? "m" : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="dahs-summary-btns">
          <button className="dahs-btn ghost" onClick={onNew}>◂ New run</button>
          <button className="dahs-btn play"  onClick={onReplay}>↻ Replay</button>
        </div>
      </div>
    </div>
  );
}

/* ─────────── tweaks panel ─────────── */
function TweaksPanel({ tweaks, setTweak, onClose }) {
  return (
    <div className="dahs-tweaks">
      <div className="dahs-tweaks-hd">
        <span>Tweaks</span>
        <button onClick={onClose} aria-label="Close">×</button>
      </div>

      <div className="dahs-tweaks-sec">
        <div className="dahs-tweaks-l">Floor layout</div>
        <div className="dahs-tweaks-seg">
          {["stacked","side","tab"].map((m) => (
            <button key={m} className={tweaks.layoutMode === m ? "on" : ""}
                    onClick={() => setTweak("layoutMode", m)}>
              {m === "stacked" ? "Stacked" : m === "side" ? "Side by side" : "Tabs"}
            </button>
          ))}
        </div>
      </div>

      <div className="dahs-tweaks-sec">
        <div className="dahs-tweaks-l">Figure annotations</div>
        <div className="dahs-tweaks-seg">
          <button className={tweaks.showAnnotations ? "on" : ""}
                  onClick={() => setTweak("showAnnotations", true)}>On</button>
          <button className={!tweaks.showAnnotations ? "on" : ""}
                  onClick={() => setTweak("showAnnotations", false)}>Off</button>
        </div>
      </div>

      <div className="dahs-tweaks-sec">
        <div className="dahs-tweaks-l">Density</div>
        <div className="dahs-tweaks-seg">
          {["compact","comfortable"].map((m) => (
            <button key={m} className={tweaks.density === m ? "on" : ""}
                    onClick={() => setTweak("density", m)}>
              {m === "compact" ? "Compact" : "Comfortable"}
            </button>
          ))}
        </div>
      </div>

      <div className="dahs-tweaks-foot">
        Persisted on save — changes survive reload.
      </div>
    </div>
  );
}

window.DAHSApp = DAHSApp;
