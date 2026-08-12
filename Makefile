# DAHS — Disruption-Aware Heuristic Scheduling
#
# Usage on Windows: install `make` (e.g. `choco install make`) or use Git-Bash /
# WSL. PowerShell users can run the `python -m <module>` lines directly.
#
# The pipeline is five ordered stages. Each depends on the artifacts of the one
# before, and Stage 1 must precede Stage 2 because it settles the rule pool and
# the ATC/COVERT look-ahead scales that labelling consumes.
#
#   stage1-*  calibrate rules, screen the pool, diagnostics   (light)
#   stage2    label the corpus                                (the expensive run)
#   stage3    regime layer + ranker + calibrator              (heavy)
#   stage4-*  baselines, including the rolling-horizon teacher
#   stage5-*  scenarios, robustness, misspecification, sensitivity
#
# NOTE. The submitted Makefile invoked `experiments.run_pilot`,
# `experiments.generate_data` and Hydra-style `--config-name=` overrides. None of
# those modules or flags existed; `make pilot`, `make data` and `make train` were
# all broken. The targets below are the real entry points.

PY ?= python
RUN_ID ?= dev

.PHONY: help install gate test clean clean-stale preflight smoke \
        stage1 stage1-calibrate stage1-screen stage1-diversity \
        stage1-perishability stage1-budget \
        stage2 stage2-smoke stage2-tau1 \
        stage3 stage3-smoke stage3-tau1 tau1 \
        stage4-static stage4-teacher stage4-fqi stage4-rl-sensitivity \
        stage5-scenarios stage5-robustness stage5-misspecification \
        stage5-sensitivity stage5-weights \
        e2-stats e3-summary e5-reliability e5-shap paper-figures

help:
	@echo "DAHS pipeline (run stages in order):"
	@echo "  install                  Install package + dev deps (editable)"
	@echo "  gate                     Import smoke test"
	@echo "  preflight                Compile + import every module (~2s)  [DO THIS FIRST]"
	@echo "  smoke                    preflight + tests + tiny label/train run"
	@echo ""
	@echo "  stage1-calibrate         Fit ATC/COVERT look-ahead scales (R1 4.c)"
	@echo "  stage1-screen            Score + screen the candidate pool (R1 4.a/b/d)"
	@echo "  stage1-diversity         Win rate across the state space (R1 4.e)"
	@echo "  stage1-perishability     Is perishability decision-relevant? (R1 1.d)"
	@echo "  stage1-budget            A priori interval-step budget (R3 1)"
	@echo "  stage1                   All of the above, in order"
	@echo ""
	@echo "  stage2                   Label the corpus  [THE EXPENSIVE RUN]"
	@echo "  stage2-smoke             3 train + 2 test shifts, end-to-end check"
	@echo "  stage3                   Regime + ranker + calibrator"
	@echo "  stage3-smoke             1 HP combo, fast end-to-end check"
	@echo "  tau1                     tau=1 arm = snapshot_xgb  [REQUIRED before stage 5]"
	@echo ""
	@echo "  stage4-static            Evaluate every deployed rule standalone"
	@echo "  stage4-teacher           Rolling-horizon MPC, the distillation teacher (R2 6)"
	@echo "  stage4-fqi               Offline FQI: hpsearch + eval (R1 6.b)"
	@echo "  stage4-rl-sensitivity    PPO hyperparameter grid + FQI coverage (R1 6.b)"
	@echo ""
	@echo "  stage5-scenarios         Scenario sweep + paired stats"
	@echo "  stage5-robustness        Untuned arrival x SLA grid"
	@echo "  stage5-misspecification  Label nominal, evaluate perturbed (R2 5)"
	@echo "  stage5-weights           Objective-weight sensitivity (R1 6.c)"
	@echo "  stage5-sensitivity       t_min and arrival-noise sweeps"
	@echo ""
	@echo "  test                     Run pytest"
	@echo "  clean-stale              Delete pre-revision labels/models/results  [DO THIS FIRST]"
	@echo "  clean                    Empty runs/, results/, figures/ (keeps dirs)"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

# --- Preflight + smoke: run this BEFORE committing to the campaign ----------
# Ordered cheapest-first, so a break costs seconds rather than hours:
#   preflight  compiles and imports every module        (~2 s)
#   test       the unit suite                           (~1 min)
#   stage2     3 train + 2 test shifts, provisional     (~1 min)
#   stage3     1 HP combo on that tiny corpus           (~1 min)
preflight:
	$(PY) scripts/preflight.py

smoke: preflight test stage2-smoke stage3-smoke
	@echo ""
	@echo "SMOKE PASSED — preflight, unit tests, labelling and training all ran."
	@echo "Next: python -m experiments.compute_budget measure --n-shifts 3"

gate:
	$(PY) -c "import xgboost, sklearn, shap, stable_baselines3, omegaconf, joblib; print('ok')"

# --- Stage 1: rule calibration and screening -------------------------------
stage1-calibrate:
	$(PY) -m experiments.calibrate_rules calibrate

stage1-screen:
	$(PY) -m experiments.calibrate_rules screen

stage1-diversity:
	$(PY) -m experiments.calibrate_rules diversity

stage1-perishability:
	$(PY) -m experiments.perishability_diagnostic

stage1-budget:
	$(PY) -m experiments.compute_budget analytic

stage1: stage1-budget stage1-calibrate stage1-screen stage1-diversity stage1-perishability

# --- Stage 2: labelling ----------------------------------------------------
stage2:
	$(PY) -m experiments.generate_labels --run-id $(RUN_ID)

# Standalone: runs on a freshly cleaned repo, BEFORE Stage 1 has fitted the
# ATC/COVERT scales. Its labels are provisional and stamped as such; the point
# is to prove the five stages are connected, not to produce anything reportable.
# Writes to data/smoke/ so it cannot clobber the real corpus.
stage2-smoke:
	$(PY) -m experiments.generate_labels --n-train 3 --n-test 2 --run-id smoke \
	  --allow-provisional-scales \
	  --train-out data/smoke/train.parquet --test-out data/smoke/test.parquet

# --- Stage 3: model --------------------------------------------------------
stage3:
	$(PY) -m experiments.train_ranker --run-id phase4

stage3-smoke:
	$(PY) -m experiments.train_ranker --smoke --skip-cv-cal --run-id smoke \
	  --train-path data/smoke/train.parquet --test-path data/smoke/test.parquet

# --- The tau=1 arm ---------------------------------------------------------
# `snapshot_xgb` IS this model. It is in the method list of e2, e8 and a2, and
# it is a reference line in the data-efficiency figure — but nothing built it,
# so the campaign failed at Stage 5 with a missing runs/phase4_tau1 after
# several hours of work. It needs its OWN labelling pass: tau changes the
# estimator, so no transform of the tau=4 parquet produces it.
# Cost: ~1.55M interval-steps, about a quarter of the tau=4 pass.
stage2-tau1:
	$(PY) -m experiments.generate_labels --tau 1 \
	  --train-out data/tau1/train.parquet --test-out data/tau1/test.parquet

stage3-tau1:
	$(PY) -m experiments.train_ranker --run-id phase4_tau1 \
	  --train-path data/tau1/train.parquet \
	  --test-path data/tau1/test.parquet --skip-cv-cal

tau1: stage2-tau1 stage3-tau1

# --- Stage 4: baselines ----------------------------------------------------
stage4-static:
	$(PY) -m experiments.evaluate --method fifo
	$(PY) -m experiments.evaluate --method edd
	$(PY) -m experiments.evaluate --method fefo
	$(PY) -m experiments.evaluate --method wspt
	$(PY) -m experiments.evaluate --method atc

stage4-teacher:
	$(PY) -m experiments.evaluate --method rolling_mpc --verbose

stage4-fqi:
	$(PY) -m experiments.e9_offline_fqi hpsearch
	$(PY) -m experiments.e9_offline_fqi eval

stage4-rl-sensitivity:
	$(PY) -m experiments.rl_sensitivity ppo
	$(PY) -m experiments.rl_sensitivity coverage

# --- Stage 5: scenarios, robustness, sensitivity ---------------------------
stage5-scenarios:
	$(PY) -m experiments.e2_main eval --scenario balanced --verbose

stage5-robustness:
	$(PY) -m experiments.e8_robustness_grid eval
	$(PY) -m experiments.e8_robustness_grid summary

stage5-misspecification:
	$(PY) -m experiments.misspecification run
	$(PY) -m experiments.misspecification summary

stage5-weights:
	$(PY) -m experiments.e4_sensitivity weights

stage5-sensitivity:
	$(PY) -m experiments.e4_sensitivity t_min
	$(PY) -m experiments.e4_sensitivity arrival_noise

# --- Reporting -------------------------------------------------------------
e2-stats:
	$(PY) -m experiments.e2_main stats --scenario default --baseline ours

e3-summary:
	$(PY) -m experiments.e3_ablations summary

e5-reliability:
	$(PY) -m experiments.e5_calibration reliability

e5-shap:
	$(PY) -m experiments.e5_calibration shap

paper-figures: stage1-diversity e2-stats e3-summary stage5-sensitivity e5-reliability e5-shap

test:
	$(PY) -m pytest

# Everything committed under data/, runs/ and results/ predates the revision:
# the deleted objective, the four-rule pool, the completed-orders-only KPI
# schema, and a test-seed block that only PARTIALLY overlaps the current one
# (the calibration block was inserted between train and test, shifting it by 30).
# Comparing new numbers against those files is not conservative, it is wrong —
# and a paired test across them would be silently misaligned rather than empty.
# Run this once before the campaign.
clean-stale:
	@$(PY) -c "import shutil, glob, os; \
[os.remove(p) for p in glob.glob('data/*.parquet')]; \
[os.remove(p) for p in glob.glob('data/*.npz')]; \
[os.remove(p) for p in glob.glob('data/label_meta.json')]; \
[shutil.rmtree(d, ignore_errors=True) for d in glob.glob('data/e3_*') + glob.glob('data/tau1') + glob.glob('data/e4_*') + glob.glob('data/smoke')]; \
[shutil.rmtree(d, ignore_errors=True) or os.makedirs(d, exist_ok=True) for d in ('runs','results','figures')]; \
print('removed pre-revision labels, models and results')"

clean:
	@$(PY) -c "import shutil, os; \
[shutil.rmtree(d, ignore_errors=True) or os.makedirs(d, exist_ok=True) for d in ('runs','results','figures')]"
