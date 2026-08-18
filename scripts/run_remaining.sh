#!/usr/bin/env bash
# Remaining campaign after commit 39984fb (Stages 2-4 artifacts present).
# Run from the repo root, lockfile venv activated:
#   source .venv/bin/activate
#   bash scripts/run_remaining.sh
#
# Do NOT re-run Stage 1 or Stage 2. Do re-run FQI after the observe-once logger fix.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then
  PY=.venv/bin/python
elif [[ -x .venv/Scripts/python.exe ]]; then
  PY=.venv/Scripts/python.exe
else
  PY=python
fi

step() {
  local title="$1"; shift
  echo
  echo "########## ${title} ##########  $(date +%H:%M:%S)"
  "$PY" "$@"
}

[[ -f data/label_meta.json ]] || { echo "ABORT: data/label_meta.json missing — copy Stages 2-4 artifacts"; exit 1; }
[[ -f runs/phase4/model.json ]] || { echo "ABORT: runs/phase4/model.json missing"; exit 1; }
[[ -f runs/phase4_tau1/model.json ]] || { echo "ABORT: runs/phase4_tau1/model.json missing — need make tau1 artifacts"; exit 1; }

step "FQI hpsearch" -m experiments.e9_offline_fqi hpsearch --n-jobs=-1
step "FQI eval"     -m experiments.e9_offline_fqi eval --n-jobs=-1
# Interim grid vs whatever E8 is on disk. Stage 5 re-runs this after e8 summary.
step "FQI E8 grid"  -m experiments.e9_offline_fqi robustness_grid --n-jobs=-1

step "snapshot_xgb default" -m experiments.evaluate --method snapshot_xgb --n-jobs=-1

step "data_efficiency ours" -m experiments.e2_main data_efficiency --n-jobs=-1
step "data_efficiency fqi"  -m experiments.e9_offline_fqi data_efficiency --n-jobs=-1
step "e9 summary"           -m experiments.e9_offline_fqi summary
step "fig_data_efficiency"  -m experiments.fig_data_efficiency

step "e2 stats default" -m experiments.e2_main stats --scenario default --baseline ours
step "e2 low_load"      -m experiments.e2_main eval --scenario low_load --n-jobs=-1
step "e2 balanced"      -m experiments.e2_main eval --scenario balanced --n-jobs=-1
step "e2 high_load"     -m experiments.e2_main eval --scenario high_load_perish --n-jobs=-1

echo
echo "REMAINING.SH block A complete. Continue with Stage 5 sensitivity in RUN_CAMPAIGN.md"
