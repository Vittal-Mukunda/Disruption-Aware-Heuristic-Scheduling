# Remaining campaign after commit 39984fb (Stages 2-4 artifacts present).
# Run from the repo root, with the lockfile venv activated:
#   .\.venv\Scripts\Activate.ps1
#   powershell -File scripts\run_remaining.ps1
#
# Do NOT re-run Stage 1 or Stage 2. Do re-run FQI after the observe-once logger fix.

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

if (-not (Test-Path "data\label_meta.json")) {
    Write-Host "ABORT: data/label_meta.json missing — copy Stages 2-4 artifacts, do not start from a bare clone of code-only."
    exit 1
}
if (-not (Test-Path "runs\phase4\model.json")) {
    Write-Host "ABORT: runs/phase4/model.json missing"
    exit 1
}
if (-not (Test-Path "runs\phase4_tau1\model.json")) {
    Write-Host "ABORT: runs/phase4_tau1/model.json missing"
    exit 1
}

function Invoke-Step([string]$Title, [string[]]$Cmd) {
    Write-Host "`n########## $Title ##########  $(Get-Date -Format 'HH:mm:ss')"
    & $py @Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ABORTED AT: $Title  rc=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# --- FQI retrain: logger was double-observing; cache stamp now invalidates ---
Invoke-Step "FQI hpsearch" @("-m", "experiments.e9_offline_fqi", "hpsearch", "--n-jobs=-1")
Invoke-Step "FQI eval"     @("-m", "experiments.e9_offline_fqi", "eval", "--n-jobs=-1")
# Interim grid vs whatever E8 is on disk. Stage 5 re-runs this after e8 summary.
Invoke-Step "FQI E8 grid"  @("-m", "experiments.e9_offline_fqi", "robustness_grid", "--n-jobs=-1")

# --- Default-scenario snapshot_xgb row (tau1 trained, never evaluated on default) ---
# --n-jobs=-1 (equals form) so PowerShell cannot eat -1 as a switch.
Invoke-Step "snapshot_xgb default" @("-m", "experiments.evaluate", "--method", "snapshot_xgb", "--n-jobs=-1")

# --- Stage 4b: sample-efficiency (central figure). Budgets are 25..250 ---
Invoke-Step "data_efficiency ours" @("-m", "experiments.e2_main", "data_efficiency", "--n-jobs=-1")
Invoke-Step "data_efficiency fqi"  @("-m", "experiments.e9_offline_fqi", "data_efficiency", "--n-jobs=-1")
Invoke-Step "e9 summary"           @("-m", "experiments.e9_offline_fqi", "summary")
Invoke-Step "fig_data_efficiency"  @("-m", "experiments.fig_data_efficiency")

# --- Stage 5 scenarios (include low_load; DAHS/snapshot honour --n-jobs) ---
Invoke-Step "e2 stats default" @("-m", "experiments.e2_main", "stats", "--scenario", "default", "--baseline", "ours")
Invoke-Step "e2 low_load"      @("-m", "experiments.e2_main", "eval", "--scenario", "low_load", "--n-jobs=-1")
Invoke-Step "e2 balanced"      @("-m", "experiments.e2_main", "eval", "--scenario", "balanced", "--n-jobs=-1")
Invoke-Step "e2 high_load"     @("-m", "experiments.e2_main", "eval", "--scenario", "high_load_perish", "--n-jobs=-1")

Write-Host "`nREMAINING.PS1 block A complete. Continue with Stage 5 sensitivity in RUN_CAMPAIGN.md"
