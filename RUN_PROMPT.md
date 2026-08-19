# Remaining-run prompt (post-campaign)

The full revision campaign is done on `ccf0240`. Do **not** wipe, do **not**
re-run Stages 1–5, do **not** run `scripts/clean_stale.py` or
`scripts/run_remaining.ps1`.

Paste everything between the `=====` lines into the agent on the **campaign
machine** (`C:\CAOR` on the i9 — keep it off OneDrive). `git pull` first.

Budget on that machine: **~3 h** for the M-sweep (required), **~30–60 min** for
`compute_budget measure` + `scaling` (should), **~6 h more** if you also run
theta (optional; skip unless you want a theta paragraph).

M=20 already exists as the main run. M=1 is also the `single_sample_rollout`
ablation — do not run that ablation separately.

---

```text
=====================================================================
You are finishing the leftover result producers for CAOR-D-26-01812 (DAHS).
The full campaign is already on main (commit ccf0240, CAMPAIGN_REPORT.md).
Your job is to run ONLY the leftover experiments, commit the artifacts, and
push. Do not redesign anything, do not tune anything, do not re-run Stages 1–5.

READ FIRST
  CAMPAIGN_REPORT.md  — what already ran, what is still open (§6.1)
  RUN_CAMPAIGN.md §6  — campaign is COMPLETE; remaining recipe sweeps only
  this file           — the operating procedure

SETUP
- Work in the existing clone. git pull origin main first.
- Python 3.12 only. Reuse the existing .venv; do not recreate it unless
  `python -c "import xgboost, sklearn"` fails.
- Windows: .venv\Scripts\python.exe
- Confirm you are NOT on a OneDrive-synced path. If you are, STOP and move
  the clone (OneDrive previously corrupted a wipe).

DO NOT
- Do not run scripts/clean_stale.py or scripts/run_remaining.ps1.
- Do not re-run Stage 1, 2, 3, 4, 4b, 5, 5b, or the existing E3 retrains.
- Do not edit config.yaml.
- Do not fill TBD-rerun markers in paper/manuscript.md.
- Do not parallelise LinUCB (you will not be evaluating it).
- Do not skip M=1: it is the single-sample ablation as well as a sweep cell.

GATE (cheap; stop on failure)
      .venv\Scripts\python.exe scripts\preflight.py
      .venv\Scripts\python.exe -m pytest -q
Expect 159 passed, 1 skipped (FEFO-not-in-pool). Any failure or extra skip:
stop and report.
      .venv\Scripts\python.exe -c "from omegaconf import OmegaConf; c=OmegaConf.load('config.yaml'); print(list(c.heuristics.pool), c.heuristics.atc_lookahead_k, c.heuristics.covert_lookahead_k)"
Must print exactly:
      ['EEDD', 'COVERT', 'MS', 'ATC', 'MDD', 'EDD'] 3.0 4.0

A. MUST — M-sweep, M in {1, 5, 10, 40}  (~3 h on the i9)
M=20 is the committed main run; do not relabel it. After M=1, copy that
eval parquet into E3 under the ablation name (evaluate always writes
ours.parquet; e3 summary keys off the filename stem).

PowerShell (campaign machine is Windows):

      foreach ($M in 1,5,10,40) {
        python -m experiments.generate_labels --n-samples $M --n-jobs -1 `
          --train-out data/e4_M$M/train.parquet --test-out data/e4_M$M/test.parquet
        python -m experiments.train_ranker --run-id e4_M$M --skip-cv-cal `
          --train-path data/e4_M$M/train.parquet --test-path data/e4_M$M/test.parquet
        python -m experiments.evaluate --method ours --n-jobs -1 `
          --run-dir runs/e4_M$M --results-dir results/E4/n_samples/M_$M
        git add data/e4_M$M runs/e4_M$M results/E4/n_samples/M_$M
        git commit -m "E4 n_samples M=$M"
        git push origin main
      }
      Copy-Item results\E4\n_samples\M_1\ours.parquet results\E3\single_sample_rollout.parquet
      python -m experiments.e3_ablations summary
      git add results/E3
      git commit -m "E3 single_sample_rollout from M=1 plus regenerated summary"
      git push origin main

Each generate_labels pass must print "median train row entropy = ... -> OK".
If it prints OUT OF BAND, stop and report; do not pass --force-out-of-band.

Commit and push after each M so a crash at M=40 does not cost M=1..10.

B. SHOULD — compute budget  (~30–60 min)
      python -m experiments.compute_budget analytic
      python -m experiments.compute_budget measure --n-shifts 3 --machine "Intel Core i9-14900K, 24c/32t"
      python -m experiments.compute_budget scaling --n-shifts 5
These write results/E12_compute/{measured_throughput.json, pool_scaling_analytic.parquet, successive_halving.json}.
results/E12_compute/latency.json and label_budget.json are already committed
from the eval parquets; do not overwrite them.

C. OPTIONAL — theta sweep  (~6 h; skip unless you want a theta paragraph)
Nominal multiple 2.2 is the main run. For each other value in
{1.5, 2.0, 3.0, 4.0}, using p instead of a dot in paths (Windows/git):

      foreach ($m in 1.5,2.0,3.0,4.0) {
        $tag = "$m" -replace '\.','p'
        python -m experiments.generate_labels --theta-uniform-multiple $m --n-jobs -1 `
          --train-out data/e4_theta_m$tag/train.parquet --test-out data/e4_theta_m$tag/test.parquet
        python -m experiments.train_ranker --run-id e4_theta_m$tag `
          --train-path data/e4_theta_m$tag/train.parquet --test-path data/e4_theta_m$tag/test.parquet
        python -m experiments.evaluate --method ours --n-jobs -1 `
          --run-dir runs/e4_theta_m$tag --results-dir results/E4/theta/m_$tag
        git add data/e4_theta_m$tag runs/e4_theta_m$tag results/E4/theta
        git commit -m "E4 theta multiple=$m"
        git push origin main
      }

ASSEMBLE (after A, and C if run)
There is no n_samples / theta summary driver. After the eval parquets exist,
write results/E4/n_samples_summary.parquet (and theta_summary.parquet if C ran)
with the same schema as results/E4/tau_summary.parquet: knob, value, metric,
point, ci_lo, ci_hi, n, n_resamples. Metrics: service_failure_rate,
composite_cost, mean_tardiness. Include the M=20 / theta=2.2 cell by copying
the existing results/ours.parquet numbers. Plot
figures/E4/n_samples_{metric}.{png,pdf} the same way e4_sensitivity plots tau.
If you cannot match the schema, leave the per-M parquets and say so in the
report; do not invent a different table.

WHAT TO WATCH, AND REPORT RATHER THAN FIX
1. Entropy band on each new labelling pass — abort if OUT OF BAND.
2. M-sweep curve: if M=1 matches M=20, the multi-sample machinery bought
   nothing and that is the result. Do not drop M=1.
3. Do not compare new M=20 numbers to the committed ours.parquet; use the
   committed file as the M=20 cell.

FINAL REPORT
Append a section "## Remaining-run addendum" to CAMPAIGN_REPORT.md:
- wall-clock per M, plus measure/scaling (and theta if run)
- for each M: median entropy, in-band?, composite_cost, service_failure_rate
- whether single_sample_rollout is in results/E3 and e3_summary was regenerated
- paths of n_samples_summary / figures, or an explicit "not assembled"
- anything that failed

Commit the new data/, runs/, results/, figures/, and the addendum. Push origin
main. Do not open a PR. Do not edit the manuscript.
=====================================================================
```

---

## Why this is all that is left

Stages 1–5, tau∈{1,2,3,4}, data-efficiency, Olist, and the four retrain
ablations already ran. Three commands exited 2 **on purpose** (they print a
recipe): `e4_sensitivity n_samples`, `e4_sensitivity theta`, and
`e3_ablations relabel single_sample_rollout`. The first two are this prompt;
the third is M=1 of the M-sweep.

`compute_budget measure` / `scaling` were never in the campaign stage list
but they are the producers for R3.1 / R3.2 (throughput, successive-halving
agreement). Latency is already extracted into `results/E12_compute/latency.json`.
