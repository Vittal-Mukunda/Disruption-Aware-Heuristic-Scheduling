# Remaining-run prompt (post-campaign)

The full revision campaign, the M-sweep, and `compute_budget measure` /
`scaling` are done on `main`. Do **not** wipe, do **not** re-run Stages 1–5
labelling/training, do **not** run `scripts/clean_stale.py` or
`scripts/run_remaining.ps1`, do **not** widen `k_grid`, do **not** run theta,
do **not** relabel.

Paste everything between the `=====` lines into the agent on the **campaign
machine** (`C:\CAOR` on the i9 — keep it off OneDrive). `git pull` first.

Every leftover eval below is **mandatory**. Terminal admit (A) changes `|A|`
and every KPI table, so it runs first. After A, every manuscript-cited eval
that calls `evaluate.run_shift` / `run_shift_env_aware` is stale and is
re-run. After all of it, update the manuscript from the live artifacts and
push.

Git commits on that machine: author **Vittal Mukunda**
`<vittal.muku@gmail.com>` only. Do not add a `Co-authored-by` trailer.
Paper YAML authors stay Vittal (corresponding), Atharva Somani, Pranjal
Malaiya — do not change the author list.

The committed default is `sim.terminal_admit: false` (live tables, mean
`|A|=767`). This run **sets it to true** and **leaves it true** so the
revised paper matches the new protocol.

---

```text
=====================================================================
You are finishing the leftover completeness evals for CAOR-D-26-01812 (DAHS).
ALL of A–K are mandatory. Do not skip any. Do not redesign, do not tune
beyond the stated overlays, do not re-run Stages 1–5 labelling/training,
do not run the M-sweep labelling or theta again.

READ FIRST
  CAMPAIGN_REPORT.md
  this file
  config.yaml  (sim.terminal_admit)
  experiments/evaluate.py
  simulation/warehouse_env.py  (admit_if_shift_complete, run_with_policy)
  experiments/e8_robustness_grid.py  (METHODS)
  experiments/rl_sensitivity.py
  baselines/rolling_horizon_mpc.py
  baselines/ppo_fair.py  (WarehouseGymEnv)

SETUP
- Existing clone at C:\CAOR. git pull origin main first.
- Confirm config.yaml contains sim.terminal_admit (gated helper already in
  warehouse_env.py / evaluate.py). If that key is missing, STOP and pull again.
- Python 3.12. Reuse .venv.
- Windows: .venv\Scripts\python.exe
- Confirm you are NOT on a OneDrive-synced path. If you are, STOP.

DO NOT
- Do not run scripts/clean_stale.py or scripts/run_remaining.ps1.
- Do not relabel or retrain the ranker.
- Do not add an unconditional _admit() at the end of run_with_policy.
  Truncated rollouts (n_steps < remaining) must not admit extra orders.
  The committed helper admit_if_shift_complete() is already gated on
  interval_idx >= n_intervals AND sim.terminal_admit.
- Do not use `e2_main eval --scenario default` for Table 6. That writes
  results/scenario_default/. Table 6 is results/<method>.parquet via
  python -m experiments.evaluate --method …
- Do not overwrite config.yaml permanently except terminal_admit (leave
  true) and the ATC k overlay, which MUST be reverted to 3.0. After the
  run, committed config must print:
      ['EEDD', 'COVERT', 'MS', 'ATC', 'MDD', 'EDD'] 3.0 4.0
  and sim.terminal_admit: true
- Do not parallelise LinUCB.
- Do not run `e2_main stats --methods fifo ours` without --out.
- Do not add Co-authored-by on commits. Author is Vittal Mukunda
  <vittal.muku@gmail.com>.
- Do not change paper author names or order.
- Do not add a generative-AI declaration.

GATE (cheap; stop on failure)
      .venv\Scripts\python.exe scripts\preflight.py
      .venv\Scripts\python.exe -m pytest tests/test_simulation.py tests/test_phase6.py -q
      .venv\Scripts\python.exe -c "from omegaconf import OmegaConf; c=OmegaConf.load('config.yaml'); print(list(c.heuristics.pool), c.heuristics.atc_lookahead_k, c.heuristics.covert_lookahead_k, bool(c.sim.get('terminal_admit', False)))"
Must print the pool and 3.0 4.0. terminal_admit may still be false before A.

A. MUST — flip terminal_admit, then re-eval every Table 6 method
In config.yaml set:

      sim.terminal_admit: true

Leave it true after the run. Do not invent a second admit path.

Then re-evaluate EVERY Table 6 method on the 50 test shifts, writing to the
canonical results/<method>.parquet paths (this is the new live table):

      foreach ($m in @(
        'eedd','covert','ms','atc','mdd','edd','fifo','wspt','fefo',
        'linucb','rolling_mpc','greedy_mpc','snapshot_xgb',
        'ppo_fair','offline_fqi','ours'
      )) {
        .venv\Scripts\python.exe -m experiments.evaluate --method $m --n-jobs -1
      }
LinUCB ignores --n-jobs (serial by design). Frozen policies/rankers: do not
retrain. Then:

      .venv\Scripts\python.exe -m experiments.e2_main stats --scenario default

Mean |A| should move from 767 toward the Poisson mean 1.65*480=792.

A2. MUST — Table 7 scenarios (same method list; driver exists)
Unconditional. Do not skip if parquets already exist.

      foreach ($s in @('low_load','balanced','high_load_perish')) {
        .venv\Scripts\python.exe -m experiments.e2_main eval --scenario $s --n-jobs -1
        .venv\Scripts\python.exe -m experiments.e2_main stats --scenario $s
      }

This refreshes results/scenario_*/ , results/E2/<scenario>_stats.parquet,
figures, and the high_load_perish WSPT claim.

B. MUST — Always-ATC at standalone k=1.5
Table 6 Always-ATC is portfolio k=3.0. `evaluate --method atc` does NOT
accept a k overlay. Do NOT use the bare CLI: it would write k=3.0 into
results/E_atc_k1p5.

      .venv\Scripts\python.exe -c "
from omegaconf import OmegaConf, open_dict
from pathlib import Path
from experiments.evaluate import canonical_test_seeds, evaluate_policy, _build_policy
cfg = OmegaConf.load('config.yaml')
with open_dict(cfg):
    cfg.heuristics.atc_lookahead_k = 1.5
assert float(cfg.heuristics.atc_lookahead_k) == 1.5
seeds = canonical_test_seeds(cfg)
policy, env_aware = _build_policy('atc', None)
assert env_aware is False
out = Path('results/E_atc_k1p5')
evaluate_policy('atc', policy, seeds, cfg, results_dir=out, save=True, n_jobs=-1)
print('ATC k in this eval:', float(cfg.heuristics.atc_lookahead_k))
"
Do not overwrite results/atc.parquet (that file is k=3.0 after A).
Restore config.yaml ATC k=3.0 if you mutated the file; the snippet above
must be in-memory only. Confirm the gate print is still 3.0 4.0.

C. MUST — E8 add Always-COVERT
In experiments/e8_robustness_grid.py set
      METHODS = ["ours", "greedy_mpc", "snapshot_xgb", "eedd", "covert"]
Re-run the 12-cell grid (frozen rankers; teachers replan with true cell
dynamics — do not pin them to the default cell):

      .venv\Scripts\python.exe -m experiments.e8_robustness_grid eval --n-jobs -1
      .venv\Scripts\python.exe -m experiments.e8_robustness_grid summary

Keep the sentence that DAHS wins 0/12 among the original four methods if
that remains true; also report COVERT's cell wins.

D. MUST — teachers at label M=20
config.yaml baselines.rolling_horizon_mpc.n_samples is 5. Overlay n_samples=20
in memory, do not commit n_samples=5 being changed.

      .venv\Scripts\python.exe -c "
from omegaconf import OmegaConf, open_dict
from pathlib import Path
from experiments.evaluate import canonical_test_seeds, evaluate_policy_env_aware
from baselines.rolling_horizon_mpc import make_rolling_horizon_mpc, make_greedy_mpc_policy
cfg = OmegaConf.load('config.yaml')
with open_dict(cfg):
    cfg.baselines.rolling_horizon_mpc.n_samples = 20
g = make_greedy_mpc_policy(cfg)
r = make_rolling_horizon_mpc(cfg, n_samples=20)
assert int(g.n_samples) == 20 and int(r.n_samples) == 20
seeds = canonical_test_seeds(cfg)
out = Path('results/E_teacher_M20')
evaluate_policy_env_aware('greedy_mpc', g, seeds, cfg, results_dir=out, save=True, n_jobs=-1)
evaluate_policy_env_aware('rolling_mpc', r, seeds, cfg, results_dir=out, save=True, n_jobs=-1)
"
Do not replace results/greedy_mpc.parquet or results/rolling_mpc.parquet
(those stay the M=5 live Table 6 rows). Report M=20 next to them.

E. MUST — PPO HP selected on calibration, frozen for test
The existing sweep scored cells on the test shifts. Do not silently replace
results/ppo_fair.parquet.

Tags (dirs runs/ppo_sensitivity/<tag>/):
  baseline, gamma=0.9, gamma=1.0, gae_lambda=0.9, gae_lambda=0.99,
  n_steps=32, n_steps=256, ent_coef=0.01, ent_coef=0.05,
  norm(obs=True,rew=False), norm(obs=False,rew=True),
  norm(obs=True,rew=True)

Gate on ppo_fair.zip, not on ppo_meta.json alone. Sensitivity zips are
gitignored. If any tag is missing a zip on C:\CAOR, retrain THAT tag
(and only then) with train_ppo_fair into that dir. Do not score the
retrain on test; selection is on calib.

      .venv\Scripts\python.exe -c "
from omegaconf import OmegaConf
from pathlib import Path
import json
from seed import shift_corpora
from experiments.evaluate import evaluate_policy, canonical_test_seeds
from baselines.ppo_fair import load_ppo_fair, train_ppo_fair
from experiments.rl_sensitivity import _sweep_configs, PPO_BASELINE
cfg = OmegaConf.load('config.yaml')
calib = shift_corpora(cfg)['calib']
test = canonical_test_seeds(cfg)
root = Path('runs/ppo_sensitivity')
rows = []
for c in _sweep_configs(cfg):
    tag = c.pop('_tag')
    run_dir = root / tag.replace('/', '_')
    zip_path = run_dir / 'ppo_fair.zip'
    if not zip_path.exists():
        print('RETRAIN missing', tag)
        hp = {k: v for k, v in c.items() if k != '_factor'}
        train_ppo_fair(run_dir=run_dir, cfg=cfg, hyperparams=hp)
    policy = load_ppo_fair(run_dir, cfg=cfg)
    df = evaluate_policy('ppo_'+tag, policy, calib, cfg, results_dir=Path('results/E_ppo_calib_select/calib'), save=True, n_jobs=1)
    rows.append({'tag': tag, 'calib_J': float(df['composite_cost'].mean())})
    print(tag, rows[-1]['calib_J'])
best = min(rows, key=lambda r: r['calib_J'])
print('WINNER', best)
policy = load_ppo_fair(root / best['tag'].replace('/', '_'), cfg=cfg)
df_test = evaluate_policy('ppo_calib_select', policy, test, cfg, results_dir=Path('results/E_ppo_calib_select'), save=True, n_jobs=1)
import pandas as pd
dahs = float(pd.read_parquet('results/ours.parquet')['composite_cost'].mean())
base = next(r['calib_J'] for r in rows if r['tag']=='baseline')
test_J = float(df_test['composite_cost'].mean())
# gap_closed vs DAHS on TEST, using the TEST-scored baseline row from the old
# sweep is wrong; compute vs the untuned ppo_fair TEST cost after A.
ppo_fair = float(pd.read_parquet('results/ppo_fair.parquet')['composite_cost'].mean())
gap = (ppo_fair - test_J) / (ppo_fair - dahs) if ppo_fair != dahs else None
out = {
    'best_tag': best['tag'],
    'calib_cost': best['calib_J'],
    'test_cost': test_J,
    'ppo_fair_test_cost': ppo_fair,
    'dahs_test_cost': dahs,
    'gap_closed_fraction_vs_ppo_fair': gap,
    'calib_rows': rows,
}
Path('results/E11_rl_sensitivity').mkdir(parents=True, exist_ok=True)
Path('results/E11_rl_sensitivity/ppo_calib_select.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
"
Do not reuse results/E11_rl_sensitivity/per_config/ppo_*.parquet for selection
(those are test-scored).

F. MUST — eval-only refreshes (frozen rankers; no relabel)
These call evaluate after A, so their absolute J would otherwise contradict
Table 6's new |A|.

      .venv\Scripts\python.exe -m experiments.e3_ablations inference
      .venv\Scripts\python.exe -c "
from omegaconf import OmegaConf
from pathlib import Path
from experiments.evaluate import canonical_test_seeds, evaluate_policy
from baselines.ours import load_ours
cfg = OmegaConf.load('config.yaml')
seeds = canonical_test_seeds(cfg)
out = Path('results/E3')
mapping = {
    'no_regime': 'runs/e3_no_regime',
    'hard_labels': 'runs/e3_hard_labels',
    'top5_features': 'runs/e3_top5_features',
    'single_sample_rollout': 'runs/e4_M1',
}
for name, rd in mapping.items():
    policy = load_ours(rd)
    evaluate_policy(name, policy, seeds, cfg, results_dir=out, save=True, n_jobs=-1)
"
      .venv\Scripts\python.exe -m experiments.e3_ablations summary
Do not treat random_ambiguity_filter as a Table 11 arm (identity; omit).

      .venv\Scripts\python.exe -m experiments.e4_sensitivity t_min --n-jobs -1
      .venv\Scripts\python.exe -m experiments.e4_sensitivity arrival_noise --n-jobs -1
      .venv\Scripts\python.exe -m experiments.e4_sensitivity tau --n-jobs -1
      .venv\Scripts\python.exe -m experiments.e4_sensitivity weights --n-jobs -1
Re-eval M-sweep rankers (do not relabel):

      .venv\Scripts\python.exe -c "
from omegaconf import OmegaConf
from pathlib import Path
from experiments.evaluate import canonical_test_seeds, evaluate_policy
from baselines.ours import load_ours
cfg = OmegaConf.load('config.yaml')
seeds = canonical_test_seeds(cfg)
pairs = [(1,'runs/e4_M1'),(5,'runs/e4_M5'),(10,'runs/e4_M10'),(20,'runs/phase4'),(40,'runs/e4_M40')]
for M, rd in pairs:
    out = Path(f'results/E4/n_samples/M_{M}')
    evaluate_policy('ours', load_ours(rd), seeds, cfg, results_dir=out, save=True, n_jobs=-1)
"

      .venv\Scripts\python.exe -m experiments.misspecification run --n-jobs -1
      .venv\Scripts\python.exe -m experiments.misspecification summary
      .venv\Scripts\python.exe -m experiments.saturation_analysis trace
      .venv\Scripts\python.exe -m experiments.saturation_analysis dwell --scenario high_load_perish
      .venv\Scripts\python.exe -m experiments.a2_olist_arrivals eval --n-jobs -1
      .venv\Scripts\python.exe -m experiments.a2_olist_arrivals summary

Data-efficiency: re-eval existing runs, do NOT retrain.

      .venv\Scripts\python.exe -c "
import json, re
from pathlib import Path
from omegaconf import OmegaConf
from experiments.evaluate import canonical_test_seeds, evaluate_policy
from baselines.ours import load_ours
cfg = OmegaConf.load('config.yaml')
seeds = canonical_test_seeds(cfg)
root = Path('runs/data_efficiency')
out = Path('results/data_efficiency')
rows = []
for d in sorted(root.iterdir()):
    if not (d/'model.json').exists():
        continue
    tag = d.name
    m = re.match(r'ours_n(\d+)_rep(\d+)$', tag)
    if not m:
        continue
    df = evaluate_policy(tag, load_ours(d), seeds, cfg, results_dir=out, save=True, n_jobs=-1)
    rows.append({'budget': int(m.group(1)), 'rep': int(m.group(2)),
                 'service_failure_rate_mean': float(df['service_failure_rate'].mean()),
                 'composite_cost_mean': float(df['composite_cost'].mean())})
    print(tag, rows[-1]['composite_cost_mean'])
(out/'data_efficiency_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
"
      .venv\Scripts\python.exe -m experiments.fig_data_efficiency

G. MUST — latency.json from post-A parquets
      .venv\Scripts\python.exe -m experiments.compute_budget latency
Do not re-run measure/scaling.

ASSEMBLE (after A–G)
      .venv\Scripts\python.exe -m experiments.e2_main stats --scenario default
with the FULL default method list (no --methods subset). This also refreshes
figures/E2/default_forest_*.{png,pdf}.

Update paper/manuscript.md numbers that A–G change, including:
- mean |A| (no longer 767 if A worked)
- Section 3.3 last-interval convention (they ARE now in A as unserved, not dispatched)
- Table 6 from the new default_stats
- Table 7 + high_load_perish WSPT from scenario stats
- ATC k=1.5 paragraph from E_atc_k1p5
- Section 6.5 / Figure 6 from the 5-method E8 grid
- teacher M=20 numbers from E_teacher_M20
- Section 6.9: calib-selected PPO row; keep ppo_fair as the untuned row
- Table 9–11, 13, 14, data-efficiency, Olist, saturation, misspec from the
  refreshed artifacts
- latency / 176x from the new latency.json
Do not change paper authors.

Keep these exact phrase anchors (audit_reviewer_items.py):
  How the reported rates are calculated
  breach rate}_{arrived}
  f_o = T + p_o
  Hierarchical selection
  Adaptive sample allocation
  Sensitivity analysis
  gap_closed_fraction
  training wall-clock to convergence
  Rank the table by composite cost
  partially observed** Markov decision process
  We withdraw those claims
  Why $M > 1
  1600$ states before\nfiltering
  grid of the two\nstate dimensions
  §8.1–8.4 headings

Then:

      .venv\Scripts\python.exe scripts\audit_reviewer_items.py
      .venv\Scripts\python.exe scripts\build_submission.py --check

Fix any gate failure you introduced. 40/40 and READY are required.

FINAL REPORT
Append '## Completeness-eval addendum' to CAMPAIGN_REPORT.md:
- wall-clock per of A–G
- new mean arrived
- ATC k=1.5 vs k=3.0 J
- E8 cell winners including COVERT
- greedy/rolling J at M=20 vs M=5
- PPO calib-selected tag and test J
- Table 7 / WSPT high_load_perish
- latency.json DAHS vs rolling
- audit/build_submission result

Commit as Vittal Mukunda <vittal.muku@gmail.com>, no Co-authored-by.
If the Cursor wrapper injects Co-authored-by, bypass with git commit-tree.
Push origin main. Do not open a PR.
=====================================================================
```
