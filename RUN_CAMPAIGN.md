# Running the CAOR-D-26-01812 revision campaign

Everything needed to take this repository from a fresh clone to the numbers that
go into the revised manuscript.

**Budget ~16 h of wall-clock, and 20-25 h on a laptop.** Run
`python scripts/campaign_budget.py` for the per-stage table on your machine; it
computes the step counts from `config.yaml` and divides by a measured throughput.
The earlier "~7 hours on 16 cores" figure was wrong in two ways: it assumed 16
*physical* cores, and it predated the four stages added below (the data-efficiency
curves, the tau sweep, the e9 grid, and the real-data figures), which together are
about a third of the total.

Two sweeps are 45% of the cost on their own — `misspecification` (~4.4 h) and the
objective-weight sweep (~2.7 h) — and in both cases it is the `rolling_mpc`
teacher that dominates, at |H| x M x tau = 480 simulated interval-steps *per
decision*. If you need to cut the campaign down, drop `rolling_mpc` from those two
sweeps first: it is load-bearing in Section 6.11 (where the online controller's
degradation is the comparison) but not in the weight sweep, which is about whether
the *ranking* survives reweighting.

Read `REVISION_PLAN.md` §7 for what each stage answers and which reviewer
comment it serves.

---

## 0. Setup

```bash
git clone https://github.com/Vittal-Mukunda/Disruption-Aware-Heuristic-Scheduling.git CAOR
cd CAOR
python -m venv .venv                    # Python 3.10-3.12 ONLY
# Install from the LOCKFILE (bit-reproducible). Windows: .venv\Scripts\pip
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/pip install -e . --no-deps
```

## 1. Gate — do not skip

```bash
.venv/bin/python scripts/preflight.py    # ~2s: compiles + imports every module
.venv/bin/python -m pytest -q            # must exit 0
```

Then wipe **pre-revision** artifacts only if this clone still has the old
objective's `data/` / `runs/` / `results/`. If `data/label_meta.json` exists
with `tau: 4` and `provisional_scales: false`, Stages 2–4 are current — **do
not wipe**. `scripts/clean_stale.py` now refuses in that case unless you pass
`--force`.

```bash
python scripts/clean_stale.py          # Windows: .venv\Scripts\python
# make clean-stale                     # optional, if make is installed
```

**Check what `clean-stale` removes before running it.** `results/S1_calibration/`
and `results/S1_perishability/` are current-revision Stage-1 outputs and must
survive; if the target would delete them, restore them from git afterwards:

```bash
git checkout -- results/S1_calibration results/S1_perishability config.yaml
```

## 2. The campaign, in order

Stage 1 **must** precede Stage 2: labelling hard-fails if ATC/COVERT have no
fitted look-ahead scale. `apply_stage1.py` writes the fitted values back into
`config.yaml`, so there is no manual editing step.

### Stage 1 is DONE — do not re-run it

Calibration, screening, the state-space diversity grid and the perishability
diagnostic have all been run on the 30-shift calibration block under the corrected
objective and dispatcher, and their artifacts are committed:

| Artifact | Result |
|---|---|
| `results/S1_calibration/rule_calibration.json` | ATC k*=1.5 standalone / 3.0 portfolio; COVERT 4.0 / 4.0 |
| `results/S1_calibration/pool_screening.json` | retained `[EEDD, COVERT, MS, ATC, MDD, EDD]`; FIFO, WSPT, FEFO dropped |
| `results/S1_calibration/diversity_state_grid.parquet` | EEDD owns 15 of 16 cells; oracle gap 7.29 pp |
| `results/S1_perishability/pivotality_summary.json` | all three pre-registered conditions met |

`config.yaml` now carries both the fitted scales and the retained pool, so Stage 2
picks them up automatically. Stage 1 does not import `soft_label_converter`, and
`simulation/`, `regime/`, `models/`, `calibrate_rules.py` and
`perishability_diagnostic.py` are unchanged since it ran — so its results are
still valid and re-running would cost ~35 min (2.97M interval-steps, mostly the
k-sweep) for identical numbers.

**If you want it re-run anyway** for single-code-state provenance, this is the
sequence; `apply_stage1.py` must be run TWICE, once after each fitting step, and
the campaign was previously launched with only the first of those done:

```bash
.venv/bin/python -m experiments.calibrate_rules calibrate --n-jobs -1
.venv/bin/python scripts/apply_stage1.py            # writes fitted k
.venv/bin/python -m experiments.calibrate_rules screen --n-jobs -1
.venv/bin/python scripts/apply_stage1.py            # writes retained pool  <-- was missed
.venv/bin/python -m experiments.calibrate_rules diversity
.venv/bin/python -m experiments.perishability_diagnostic --n-jobs -1
```

```bash
# --- Stage 2: label the corpus (~45 min) ----------------------------------
.venv/bin/python -m experiments.generate_labels --run-id phase4 --n-jobs -1

# --- Stage 3: model (~35 min) ---------------------------------------------
.venv/bin/python -m experiments.train_ranker --run-id phase4

# --- tau=1 arm = snapshot_xgb; REQUIRED before Stage 5 (~45 min) ----------
make tau1

# --- Stage 4: baselines (~2 h) --------------------------------------------
for m in eedd covert ms atc mdd edd fifo wspt fefo; do
  .venv/bin/python -m experiments.evaluate --method $m --n-jobs -1
done
# The last three are screened OUT of the pool but are still reported as
# standalone benchmarks: Table 1 has to show what the selector is beating,
# including the rules the screen rejected.
.venv/bin/python -m experiments.evaluate --method ours --verbose --n-jobs=-1
# The two lookahead controllers are the expensive evaluations: the tau-step
# teacher costs |H|*M*tau = 480 simulated interval-steps PER DECISION, measured
# at ~32 s/shift. Serial that is ~26 min for 50 shifts; -1 cuts it to a few.
# LinUCB keeps weights across shifts and is forced serial. DAHS / snapshot_xgb
# reset per shift and honour --n-jobs.
.venv/bin/python -m experiments.evaluate --method rolling_mpc --verbose --n-jobs -1
.venv/bin/python -m experiments.evaluate --method greedy_mpc --n-jobs -1
.venv/bin/python -m experiments.evaluate --method linucb
.venv/bin/python -m experiments.evaluate --method snapshot_xgb --n-jobs -1
# FQI: if this tree already has results/offline_fqi.parquet from before the
# observe-once logger fix, DELETE data/offline_fqi_transitions.npz (or rely on
# the cache stamp `logger=observe_once`) and re-run hpsearch + eval.
.venv/bin/python -m experiments.e9_offline_fqi hpsearch --n-jobs=-1
.venv/bin/python -m experiments.e9_offline_fqi eval --n-jobs=-1
# robustness_grid compares against E8; E8 is Stage 5. The command now writes
# FQI's own 12-cell parquet even if E8 is missing. Re-run it after E8 summary.
.venv/bin/python -m experiments.e9_offline_fqi robustness_grid --n-jobs=-1
.venv/bin/python -m baselines.ppo_fair
.venv/bin/python -m experiments.evaluate --method ppo_fair
.venv/bin/python -m experiments.rl_sensitivity ppo
.venv/bin/python -m experiments.rl_sensitivity coverage

# --- Stage 4b: the sample-efficiency curves (~1.5 h) ----------------------
# THE CENTRAL FIGURE. snapshot_xgb must exist as a reference line.
.venv/bin/python -m experiments.evaluate --method snapshot_xgb --n-jobs=-1
.venv/bin/python -m experiments.e2_main data_efficiency --n-jobs=-1
.venv/bin/python -m experiments.e9_offline_fqi data_efficiency --n-jobs=-1
.venv/bin/python -m experiments.e9_offline_fqi summary
.venv/bin/python -m experiments.fig_data_efficiency

# --- Stage 5: scenarios, robustness, sensitivity (~2 h) -------------------
.venv/bin/python -m experiments.e2_main stats --scenario default --baseline ours
.venv/bin/python -m experiments.e2_main eval --scenario low_load --n-jobs -1
.venv/bin/python -m experiments.e2_main eval --scenario balanced --n-jobs -1
.venv/bin/python -m experiments.e2_main eval --scenario high_load_perish --n-jobs -1
.venv/bin/python -m experiments.e8_robustness_grid eval --n-jobs -1
.venv/bin/python -m experiments.e8_robustness_grid summary
.venv/bin/python -m experiments.e9_offline_fqi robustness_grid --n-jobs=-1
.venv/bin/python -m experiments.misspecification run --n-jobs -1
.venv/bin/python -m experiments.misspecification summary
.venv/bin/python -m experiments.e4_sensitivity weights --n-jobs -1
.venv/bin/python -m experiments.e4_sensitivity t_min --n-jobs=-1
.venv/bin/python -m experiments.e4_sensitivity arrival_noise --n-jobs=-1
# The next three PRINT A RECIPE and exit 2. That is success, not a crash:
# continue. Do not wrap Stage 5 in `set -e` / stop-on-error that treats 2 as abort.
.venv/bin/python -m experiments.e4_sensitivity theta
# The tau sweep needs a LABELLING pass per tau, not just a retrain, because
# tau changes the estimator. Both of these PRINT A RECIPE rather than running
# it -- follow the printed commands. tau=1 is already built by `make tau1`.
.venv/bin/python -m experiments.e4_sensitivity tau --n-jobs=-1
.venv/bin/python -m experiments.e4_sensitivity n_samples
.venv/bin/python -m experiments.e5_calibration reliability
.venv/bin/python -m experiments.e5_calibration shap
.venv/bin/python -m experiments.feature_analysis
.venv/bin/python -m experiments.observability_analysis
# Section 6.2's boundary-conditions analysis (R3.3) needs BOTH halves. `trace`
# gives the selection entropy, the switch rate and the blocked-switch rate across
# every scenario — the statistics that distinguish "the selector collapsed" from
# "the dwell guardrail bound". `dwell` is the causal follow-up inside the one
# scenario DAHS lost. Only `dwell` was in the campaign, so R3.3 would have been
# half-answered.
.venv/bin/python -m experiments.saturation_analysis trace
.venv/bin/python -m experiments.saturation_analysis dwell --scenario high_load_perish

# --- Stage 5b: real-data grounding (~20 min) ------------------------------
# Figures 7 and 8, and the whole of R1.5b. Needs the Olist dataset unzipped into
# `Olist Dataset/` at the repo root; if it is absent these three are the ONLY
# things that cannot run, and that must be reported rather than skipped silently.
.venv/bin/python -m experiments.fit_input_distributions
.venv/bin/python -m experiments.a_realdata_validation
# a2 takes a SUBCOMMAND. Called bare it exits on argument parsing and Figure 8
# is never produced -- which is how it was written here until it was run once.
.venv/bin/python -m experiments.a2_olist_arrivals eval
.venv/bin/python -m experiments.a2_olist_arrivals summary

# --- Ablations (~2 h; independent of the headline, can run last) ----------
.venv/bin/python -m experiments.e3_ablations inference
for a in no_regime hard_labels random_ambiguity_filter top5_features; do
  .venv/bin/python -m experiments.e3_ablations retrain $a
done
.venv/bin/python -m experiments.e3_ablations relabel single_sample_rollout   # prints recipe
.venv/bin/python -m experiments.e3_ablations summary
```

Commit and push after each stage.

---

## 3. Numbers to extract for the manuscript

| Artifact | Feeds |
|---|---|
| `results/S1_calibration/rule_calibration.json` | R1.4c — both fitted k, deployed value |
| `results/S1_calibration/pool_screening.json` | R1.4b/4d — screening table, retained pool |
| `results/S1_calibration/diversity_state_grid.parquet` | R1.4e — replaces Figure 1 |
| `results/S1_perishability/pivotality_summary.json` | R1.1d — the three pre-registered conditions |
| `data/label_meta.json` | R2.3, R3.1 — beta AND beta_mode, entropy, rollout SE, interval-steps |
| `runs/phase4/phase4_regime.json` | R1.3a — BIC sweep, K*, ARI |
| `runs/phase4/phase4_metrics.json` | CV soft-xent, ECE pre/post, argmax distributions |
| `results/*.parquet` | Table 1 — per-method KPIs |
| `results/E2/default_stats.parquet` | bootstrap CI + Wilcoxon + BH-FDR |
| `results/E3/e3_summary.parquet`, `e3_cost_summary.parquet` | ablations + R3.5 cost columns |
| `results/E4/weights_summary.parquet` | R1.6c — objective-weight sensitivity |
| `results/E8/robustness_grid_summary.parquet` | untuned-cell robustness |
| `results/E10_misspecification/misspecification.parquet` | R2.5 |
| `results/E11_rl_sensitivity/ppo_sensitivity.json` | R1.6b — does tuning close the PPO gap? |
| `results/E9/` | offline-FQI comparison, incl. its sample-efficiency curve |
| `runs/data_efficiency/`, `figures/data_efficiency/` | Figure 4 — THE central figure (Section 6.3) |
| `results/E4/tau_summary.parquet` | Table 3, Figure 5 — the rollout-horizon sweep |
| `results/A/`, `results/A2/` | Figures 7-8, and the fitted input distributions (R1.5b) |

---

## 4. Things that must be reported, not smoothed over

**The headline margin shrinks.** Counting unserved-and-overdue orders as failures
(R2.1) moves the advantage over FIFO from ~3.8x to ~1.20x on this repository's
own committed run logs. That is the correct number. Section 6, Table 1 and the
abstract must be written around it.

**The diversity gate is broken, and the grid is worse than the gate.** EEDD —
the rule that sorts on `min(sla_due, expiry_time)` — wins 65.0% of decisions on
the calibration corpus, above the project's own pre-registered 60% ceiling. Worse,
the state-space grid shows it owns **15 of 16 cells**: the best single rule wins
65.00% and the per-cell oracle only 72.29%, a gap of 7.29 points.

This is reported in Section 6.1 and it is the biggest risk to the headline. A
selector has little room over "always EEDD" at that resolution. Two things could
still rescue it — the grid oracle is a floor rather than a ceiling, since DAHS
reads 26 features and not two binned ones; and win rate is not cost, so a rule
that wins rarely on expensive states can still pay. **Watch the composite-cost
margin over EEDD-alone in Stage 4.** If DAHS does not clear EEDD by a
statistically meaningful margin on composite cost, say so and rest the paper on
the sample-efficiency and amortisation results rather than on "selection beats any
single rule".

**The label entropy band — fixed, but still check it.** This was a live risk and
it has been closed. The corrected objective makes the per-row cost spread vary by
two orders of magnitude across a shift (~0 at the first epochs to ~172 at the
last on the smoke corpus), and a single global temperature cannot serve both ends:
on the 3-shift smoke run it produced a median label entropy of 1.60 nats against a
target band of [0.45, 1.05], and the test ambiguity filter then discarded 45 of 64
states. `labeling.soft_label_converter` now tempers per row —
`softmax(-J_h(s) / (beta * sigma(s)))`, `labeling.beta_mode: per_row` — which
brings the same corpus to 0.67 nats, in band, and the filter keeps 61 of 64.

This is a method change, not a bug fix: it alters every label, and the paper
reports it as such (Section 4.3, Appendix B). `beta_mode: global` reproduces the
submitted construction if the comparison is wanted.

Stage 2 still prints the achieved median against the band. If it prints OUT OF
BAND at full scale, the process **aborts** unless you pass `--force-out-of-band`
(or `--allow-provisional-scales` for a smoke run). Do not tune around a failed
band; report it.

**Check `frac_separation_below_1se`** in `label_meta.json`. On a smoke corpus 79%
of epochs had a best/second-best gap under one pooled standard error at M=20. If
that holds at full scale the soft labels are largely noise, and the E4 `n_samples`
sweep is the evidence either way.

**Do not parallelise LinUCB.** It is an online learner whose weights persist
across shifts by design; `parallel_safe = False` enforces this and the harness
honours it over any `--n-jobs`.

---

## 5. If something breaks

`scripts/preflight.py` is the first thing to run — it imports every module and
prints a traceback per failure. Most breakage surfaces there in two seconds
rather than at hour three.

Stage 3 refuses to train without a `label_meta.json` beside its train parquet.
That is deliberate: pre-revision label files are structurally valid and would
train a plausible model on the deleted objective. Re-run Stage 2 rather than
bypassing it.

---

## 6. This clone already has Stages 1–4

Do **not** run `make clean-stale` or `make clean` here: that deletes the committed
Stage-2 labels, Stage-3 ranker, and Stage-4 parquets. Stage 1 is done. Stages 2–3
are done. Stage 4 baselines exist, but:

- `offline_fqi` must be retrained (double-`observe()` logger, now fixed).
- `results/snapshot_xgb.parquet` for the default scenario was never written
  (`evaluate --method snapshot_xgb`).
- Stages 4b and 5 (data-efficiency, low_load / balanced / high_load_perish,
  misspecification, weights, SHAP) have not run.

On Windows, from the repo root with the lockfile venv:

```powershell
.\.venv\Scripts\Activate.ps1
powershell -File scripts\run_remaining.ps1
```

That block retrains FQI, evaluates snapshot_xgb, then Stage 4b and the three
scenario evals. Continue afterwards from Stage 5 in section 2 above
(`e8_robustness_grid`, misspecification, E4, E5). `e4_sensitivity tau`,
`n_samples`, `theta`, and `e3_ablations relabel` **exit 2 on purpose** — they
print a recipe; a zero exit would look like a finished sweep.

Python on this machine is `.venv\Scripts\python.exe`, not `.venv/bin/python`.
`make` is optional; every Makefile target is a `python -m ...` line.
