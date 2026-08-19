# Remaining-run prompt (post-campaign)

The full revision campaign, the M-sweep, and `compute_budget measure` /
`scaling` are done on `main`. Do **not** wipe, do **not** re-run Stages 1–5
labelling/training, do **not** run `scripts/clean_stale.py` or
`scripts/run_remaining.ps1`, do **not** widen `k_grid`, do **not** run theta.

Paste everything between the `=====` lines into the agent on the **campaign
machine** (`C:\CAOR` on the i9 — keep it off OneDrive). `git pull` first.

Every leftover eval below is **mandatory**. Terminal admit (A) changes
`|A|` and every KPI table, so it runs first. The rest run on that environment.
After all five, update the manuscript from the live artifacts and push.

Git commits on that machine: author **Vittal Mukunda** only. Do not add a
`Co-authored-by` trailer. Paper YAML authors stay Vittal (corresponding),
Atharva Somani, Pranjal Malaiya — do not change the author list.

---

```text
=====================================================================
You are finishing the leftover completeness evals for CAOR-D-26-01812 (DAHS).
ALL of A–E are mandatory. Do not skip any. Do not redesign, do not tune
beyond the stated overlays, do not re-run Stages 1–5 labelling/training,
do not run the M-sweep or theta again.

READ FIRST
  CAMPAIGN_REPORT.md
  this file
  experiments/evaluate.py
  simulation/warehouse_env.py  (run_shift / run_with_policy / _admit)
  experiments/e8_robustness_grid.py  (METHODS)
  experiments/rl_sensitivity.py
  baselines/rolling_horizon_mpc.py  (n_samples override)

SETUP
- Existing clone. git pull origin main first.
- Python 3.12. Reuse .venv.
- Windows: .venv\Scripts\python.exe
- Confirm you are NOT on a OneDrive-synced path. If you are, STOP.

DO NOT
- Do not run scripts/clean_stale.py or scripts/run_remaining.ps1.
- Do not relabel or retrain the ranker (labels do not see a terminal admit).
- Do not overwrite config.yaml permanently. Overlays must be reverted;
  committed config must still print:
      ['EEDD', 'COVERT', 'MS', 'ATC', 'MDD', 'EDD'] 3.0 4.0
- Do not parallelise LinUCB.
- Do not run `e2_main stats --methods fifo ours` without --out.
- Do not add Co-authored-by on commits. Author is Vittal Mukunda.
- Do not change paper author names or order.

GATE (cheap; stop on failure)
      .venv\Scripts\python.exe scripts\preflight.py
      .venv\Scripts\python.exe -m pytest tests/test_phase6.py -q
      .venv\Scripts\python.exe -c "from omegaconf import OmegaConf; c=OmegaConf.load('config.yaml'); print(list(c.heuristics.pool), c.heuristics.atc_lookahead_k, c.heuristics.covert_lookahead_k)"
Must print exactly:
      ['EEDD', 'COVERT', 'MS', 'ATC', 'MDD', 'EDD'] 3.0 4.0

A. MUST — terminal admit at T, then re-eval every Table 6 method
warehouse_env currently never admits arrivals in (T-L, T]. After the last
review, t becomes T and the loop stops. Add a single terminal `_admit()`
after the shift loop, before KPIs, in every completion path:

  - experiments/evaluate.py  `run_shift`
  - experiments/evaluate.py  `run_shift_env_aware`
  - simulation/warehouse_env.py  `run_with_policy`

Do not dispatch those orders (there is no review at T). They enter A as
unserved. Remove or rewrite the comment that says do not add a terminal
admit. Mean |A| should move from 767 toward the Poisson mean 1.65*480=792.

Then re-evaluate EVERY Table 6 method on the 50 test shifts, writing to the
canonical results/<method>.parquet paths (this is the new live table):

      foreach ($m in @(
        'eedd','covert','ms','atc','mdd','edd','fifo','wspt','fefo',
        'linucb','rolling_mpc','greedy_mpc','snapshot_xgb',
        'ppo_fair','offline_fqi','ours'
      )) {
        python -m experiments.evaluate --method $m --n-jobs -1
      }
LinUCB ignores --n-jobs (serial by design). Frozen policies/rankers: do not
retrain. Then:

      python -m experiments.e2_main stats --scenario default

Also re-eval the four named scenarios if those parquets exist under
results/scenario_* (same method list, via e2_main eval). If a scenario
driver is missing, say so in the report rather than inventing one.

B. MUST — Always-ATC at standalone k=1.5
Table 6 Always-ATC is portfolio k=3.0. Overlay k=1.5 without committing it.
Evaluate method atc into a NEW directory; do not overwrite results/atc.parquet
(that file is k=3.0 after A):

      python -m experiments.evaluate --method atc --n-jobs -1 --results-dir results/E_atc_k1p5

Restore config.yaml ATC k=3.0 before the next step. Confirm the gate print
is still 3.0 4.0.

C. MUST — E8 add Always-COVERT
In experiments/e8_robustness_grid.py set
      METHODS = ["ours", "greedy_mpc", "snapshot_xgb", "eedd", "covert"]
Re-run the 12-cell grid (frozen rankers; teachers replan with true cell
dynamics — do not pin them to the default cell):

      python -m experiments.e8_robustness_grid eval --n-jobs -1
      python -m experiments.e8_robustness_grid summary

Keep the sentence that DAHS wins 0/12 among the original four methods if
that remains true; also report COVERT's cell wins.

D. MUST — teachers at label M=20
config.yaml baselines.rolling_horizon_mpc.n_samples is 5. Overlay n_samples=20
in memory (or a temp cfg), do not commit n_samples=5 being changed.
Evaluate both teachers into a NEW directory:

      python -c "
from omegaconf import OmegaConf, open_dict
from pathlib import Path
from experiments.evaluate import canonical_test_seeds, evaluate_policy_env_aware
from baselines.rolling_horizon_mpc import make_rolling_horizon_mpc, make_greedy_mpc_policy
cfg = OmegaConf.load('config.yaml')
with open_dict(cfg):
    cfg.baselines.rolling_horizon_mpc.n_samples = 20
seeds = canonical_test_seeds(cfg)
out = Path('results/E_teacher_M20')
evaluate_policy_env_aware('greedy_mpc', make_greedy_mpc_policy(cfg), seeds, cfg, results_dir=out, save=True)
evaluate_policy_env_aware('rolling_mpc', make_rolling_horizon_mpc(cfg, n_samples=20), seeds, cfg, results_dir=out, save=True)
"
Do not replace results/greedy_mpc.parquet or results/rolling_mpc.parquet
(those stay the M=5 live Table 6 rows). Report M=20 next to them.

E. MUST — PPO HP selected on calibration, frozen for test
The existing sweep scored cells on the test shifts. Do not silently replace
results/ppo_fair.parquet.

If runs/ppo_sensitivity/<tag> already exist, do NOT retrain. For each tag,
evaluate the frozen policy on the CALIBRATION block (seed.shift_corpora(cfg)['calib']).
Pick the tag with lowest mean composite_cost on calib. Evaluate that frozen
policy once on the TEST block into results/E_ppo_calib_select/.

If terminal admit (A) makes those old PPO runs incomparable and you judge
retraining necessary, retrain the sweep evaluating each cell on calib (not
test), then freeze the winner for the test eval. Write
results/E11_rl_sensitivity/ppo_calib_select.json with best_tag, calib cost,
test cost, and gap_closed_fraction versus DAHS on test.

ASSEMBLE (after A–E)
      python -m experiments.e2_main stats --scenario default
with the FULL default method list (no --methods subset).

Update paper/manuscript.md numbers that A–E change, including:
- mean |A| (no longer 767 if A worked)
- Section 3.3 last-interval convention (they ARE now in A, not dispatched)
- Table 6 from the new default_stats
- ATC k=1.5 paragraph from E_atc_k1p5
- Section 6.5 / Figure 6 from the 5-method E8 grid
- teacher M=20 numbers from E_teacher_M20
- Section 6.9: calib-selected PPO row, keep ppo_fair as the untuned test-scored row
Do not change paper authors. Then:

      python scripts/audit_reviewer_items.py
      python scripts/build_submission.py --check

Fix any gate failure you introduced. 40/40 and READY are required.

FINAL REPORT
Append "## Completeness-eval addendum" to CAMPAIGN_REPORT.md:
- wall-clock per of A–E
- new mean arrived
- ATC k=1.5 vs k=3.0 J
- E8 cell winners including COVERT
- greedy/rolling J at M=20 vs M=5
- PPO calib-selected tag and test J
- audit/build_submission result

Commit as Vittal Mukunda, no Co-authored-by. Push origin main. Do not open a PR.
=====================================================================
```
