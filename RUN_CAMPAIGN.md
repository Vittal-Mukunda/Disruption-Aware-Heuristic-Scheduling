# Running the CAOR-D-26-01812 revision campaign

Everything needed to take this repository from a fresh clone to the numbers that
go into the revised manuscript. Roughly **7 hours** on 16 cores.

Read `REVISION_PLAN.md` §7 for what each stage answers and which reviewer
comment it serves.

---

## 0. Setup

```bash
git clone https://github.com/Vittal-Mukunda/Disruption-Aware-Heuristic-Scheduling.git CAOR
cd CAOR
python -m venv .venv                    # Python 3.10-3.12 ONLY
.venv/bin/pip install -e ".[dev]"       # Windows: .venv\Scripts\pip
```

## 1. Gate — do not skip

```bash
.venv/bin/python scripts/preflight.py    # ~2s: compiles + imports every module
.venv/bin/python -m pytest -q            # must exit 0
```

Then wipe every pre-revision artifact. This is mandatory, not tidiness: the
committed `data/`, `runs/` and `results/` predate the corrected objective, and
inserting the calibration block shifted the test-seed range so old and new
results overlap on only 20 of 50 seeds. A paired comparison across them is
silently misaligned rather than obviously empty.

```bash
make clean-stale
```

## 2. The campaign, in order

Stage 1 **must** precede Stage 2: labelling hard-fails if ATC/COVERT have no
fitted look-ahead scale. `apply_stage1.py` writes the fitted values back into
`config.yaml`, so there is no manual editing step.

```bash
# --- Stage 1: calibrate + screen (~35 min) --------------------------------
.venv/bin/python -m experiments.calibrate_rules calibrate --n-jobs -1
.venv/bin/python scripts/apply_stage1.py            # writes fitted k
.venv/bin/python -m experiments.calibrate_rules screen --n-jobs -1
.venv/bin/python scripts/apply_stage1.py            # writes retained pool
.venv/bin/python -m experiments.calibrate_rules diversity
.venv/bin/python -m experiments.perishability_diagnostic --n-jobs -1
git add -A && git commit -m "Stage 1: fitted scales and screened pool"

# --- Stage 2: label the corpus (~45 min) ----------------------------------
.venv/bin/python -m experiments.generate_labels --run-id phase4 --n-jobs -1

# --- Stage 3: model (~35 min) ---------------------------------------------
.venv/bin/python -m experiments.train_ranker --run-id phase4

# --- tau=1 arm = snapshot_xgb; REQUIRED before Stage 5 (~45 min) ----------
make tau1

# --- Stage 4: baselines (~2 h) --------------------------------------------
for m in eedd covert ms atc mdd edd; do
  .venv/bin/python -m experiments.evaluate --method $m
done
.venv/bin/python -m experiments.evaluate --method ours --verbose
.venv/bin/python -m experiments.evaluate --method rolling_mpc --verbose
.venv/bin/python -m experiments.evaluate --method greedy_mpc
.venv/bin/python -m experiments.evaluate --method linucb
.venv/bin/python -m experiments.e9_offline_fqi hpsearch
.venv/bin/python -m experiments.e9_offline_fqi eval
.venv/bin/python -m baselines.ppo_fair
.venv/bin/python -m experiments.evaluate --method ppo_fair
.venv/bin/python -m experiments.rl_sensitivity ppo
.venv/bin/python -m experiments.rl_sensitivity coverage

# --- Stage 5: scenarios, robustness, sensitivity (~2 h) -------------------
.venv/bin/python -m experiments.e2_main stats --scenario default --baseline ours
.venv/bin/python -m experiments.e2_main eval --scenario balanced
.venv/bin/python -m experiments.e2_main eval --scenario high_load_perish
.venv/bin/python -m experiments.e8_robustness_grid eval --n-jobs -1
.venv/bin/python -m experiments.e8_robustness_grid summary
.venv/bin/python -m experiments.misspecification run --n-jobs -1
.venv/bin/python -m experiments.misspecification summary
.venv/bin/python -m experiments.e4_sensitivity weights --n-jobs -1
.venv/bin/python -m experiments.e4_sensitivity t_min
.venv/bin/python -m experiments.e4_sensitivity arrival_noise
.venv/bin/python -m experiments.e5_calibration reliability
.venv/bin/python -m experiments.e5_calibration shap
.venv/bin/python -m experiments.feature_analysis
.venv/bin/python -m experiments.observability_analysis
.venv/bin/python -m experiments.saturation_analysis dwell --scenario high_load_perish

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
| `data/label_meta.json` | R2.3, R3.1 — beta, entropy, rollout SE, interval-steps |
| `runs/phase4/phase4_regime.json` | R1.3a — BIC sweep, K*, ARI |
| `runs/phase4/phase4_metrics.json` | CV soft-xent, ECE pre/post, argmax distributions |
| `results/*.parquet` | Table 1 — per-method KPIs |
| `results/E2/default_stats.parquet` | bootstrap CI + Wilcoxon + BH-FDR |
| `results/E3/e3_summary.parquet`, `e3_cost_summary.parquet` | ablations + R3.5 cost columns |
| `results/E4/weights_summary.parquet` | R1.6c — objective-weight sensitivity |
| `results/E8/robustness_grid_summary.parquet` | untuned-cell robustness |
| `results/E10_misspecification/misspecification.parquet` | R2.5 |
| `results/E11_rl_sensitivity/ppo_sensitivity.json` | R1.6b — does tuning close the PPO gap? |
| `results/E9/` | offline-FQI comparison |

---

## 4. Things that must be reported, not smoothed over

**The headline margin shrinks.** Counting unserved-and-overdue orders as failures
(R2.1) moves the advantage over FIFO from ~3.8x to ~1.20x on this repository's
own committed run logs. That is the correct number. Section 6, Table 1 and the
abstract must be written around it.

**One rule may break the diversity gate.** On the calibration corpus EEDD — the
rule that sorts on `min(sla_due, expiry_time)` — wins ~65% of decisions, above
the project's own pre-registered 60% ceiling. `screen` prints the warning. Report
the top win rate; do not drop the rule to get under the gate.

**Check the label entropy band.** Stage 2 prints the median label entropy against
its target. The corrected objective makes cost scale vary ~13x across a shift, so
a single global temperature may leave labels near-uniform. If it prints OUT OF
BAND, report the number and raise it — the fix is per-row tempering
(`softmax(-J_h(s) / (beta * sigma(s)))`), which is a method change, not a bug fix.

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
