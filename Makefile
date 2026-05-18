# DAHS — Rollout-Informed LDL for Adaptive Heuristic Selection
# Usage on Windows: install `make` (e.g., via Chocolatey: `choco install make`)
# or use Git-Bash / WSL. PowerShell users can also run `python -m <module>` directly.

PY ?= python
RUN_ID ?= $(shell date +%Y%m%d-%H%M%S)

.PHONY: help install gate pilot data train eval paper-figures test clean \
        e1 e2-stats e2-eval e2-data-efficiency \
        e3-inference e3-retrain-no-regime e3-retrain-hard-labels \
        e3-retrain-random-filter e3-summary \
        e4-tmin e4-arrival e4-tau e4-theta \
        e5-reliability e5-shap

help:
	@echo "DAHS targets:"
	@echo "  install                Install package + dev deps (editable)"
	@echo "  gate                   Phase 0 import smoke test"
	@echo "  pilot                  Phase 2: 50-shift pilot heuristic-diversity check"
	@echo "  data                   Phase 3: generate 250 train + 50 test labeled shifts"
	@echo "  train                  Phase 4: train calibrated LDL ranker"
	@echo "  eval                   Phase 5: per-method shift evaluation harness"
	@echo "  e1                     Phase 6 E1: heuristic-diversity heatmap"
	@echo "  e2-stats               Phase 6 E2: bootstrap + Wilcoxon + BH-FDR on existing parquets"
	@echo "  e2-eval                Phase 6 E2: re-evaluate all methods on a named scenario"
	@echo "  e2-data-efficiency     Phase 6 E2: retrain OURS at multiple train-budget cells (HOURS)"
	@echo "  e3-inference           Phase 6 E3: no_calibration + no_switching_controller (eval-only)"
	@echo "  e3-retrain-no-regime   Phase 6 E3: retrain without GMM features (HOUR+)"
	@echo "  e3-retrain-hard-labels Phase 6 E3: retrain on one-hot labels (HOUR+)"
	@echo "  e3-retrain-random-filter Phase 6 E3: retrain with random row-drop (HOUR+)"
	@echo "  e3-summary             Phase 6 E3: aggregate ablation parquets + paired stats"
	@echo "  e4-tmin                Phase 6 E4: T_min sweep (eval-only)"
	@echo "  e4-arrival             Phase 6 E4: arrival-rate-multiplier sweep (eval-only)"
	@echo "  e4-tau                 Phase 6 E4: rollout-horizon sweep (uses pretrained; tau=2/3 need retrain)"
	@echo "  e4-theta               Phase 6 E4: confidence-filter sweep (re-labeling required)"
	@echo "  e5-reliability         Phase 6 E5: pre/post-isotonic reliability diagrams + ECE table"
	@echo "  e5-shap                Phase 6 E5: global + per-class SHAP summary"
	@echo "  paper-figures          Phase 8: regenerate all manuscript figures"
	@echo "  test                   Run pytest"
	@echo "  clean                  Remove runs/, results/, figures/ contents (keeps dirs)"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

# Phase 0 test gate (per master plan)
gate:
	$(PY) -c "import simpy, xgboost, sklearn, shap, stable_baselines3; print('ok')"

pilot:
	$(PY) -m experiments.run_pilot --config-name=pilot run_id=$(RUN_ID)

data:
	$(PY) -m experiments.generate_data --config-name=data run_id=$(RUN_ID)

train:
	$(PY) -m experiments.train_ranker --config-name=train run_id=$(RUN_ID)

eval:
	$(PY) -m experiments.evaluate --method ours --verbose

# --- Phase 6 ---
e1:
	$(PY) -m experiments.e1_diversity

e2-stats:
	$(PY) -m experiments.e2_main stats --scenario default --baseline ours

e2-eval:
	$(PY) -m experiments.e2_main eval --scenario balanced --verbose

e2-data-efficiency:
	$(PY) -m experiments.e2_main data_efficiency

e3-inference:
	$(PY) -m experiments.e3_ablations inference

e3-retrain-no-regime:
	$(PY) -m experiments.e3_ablations retrain no_regime

e3-retrain-hard-labels:
	$(PY) -m experiments.e3_ablations retrain hard_labels

e3-retrain-random-filter:
	$(PY) -m experiments.e3_ablations retrain random_ambiguity_filter

e3-summary:
	$(PY) -m experiments.e3_ablations summary

e4-tmin:
	$(PY) -m experiments.e4_sensitivity t_min

e4-arrival:
	$(PY) -m experiments.e4_sensitivity arrival_noise

e4-tau:
	$(PY) -m experiments.e4_sensitivity tau

e4-theta:
	$(PY) -m experiments.e4_sensitivity theta

e5-reliability:
	$(PY) -m experiments.e5_calibration reliability

e5-shap:
	$(PY) -m experiments.e5_calibration shap

paper-figures: e1 e2-stats e3-inference e4-tmin e4-arrival e5-reliability e5-shap

test:
	$(PY) -m pytest

clean:
	@$(PY) -c "import shutil, os; \
[shutil.rmtree(d, ignore_errors=True) or os.makedirs(d, exist_ok=True) for d in ('runs','results','figures')]"
