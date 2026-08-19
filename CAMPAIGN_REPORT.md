# CAOR-D-26-01812 — Campaign Report

Run executed against commit `045edbc` ("Fix FQI logger, unserved KPIs, and campaign
gates before a from-scratch re-run"), merged locally as `c4d8f84`.

**An earlier full campaign was run against the pre-fix tree and is superseded.** It
survives only in git history (`05db389` … `0fd99f6`). Those logs have been deleted.
The live record of *this* run is `campaign_logs/rerun.log`. **Do not cite any
number from the pre-fix campaign.**

---

## 1. Machine and environment

| | |
|---|---|
| CPU | Intel Core i9-14900K, 24 physical cores / 32 logical |
| RAM | 63.8 GB |
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.12.10 |
| Repo path | `C:\CAOR` (deliberately outside OneDrive — see §6.2) |

This is a desktop, not the "13in laptop chassis" the budget's throttling rows assume, so
the −20% / −35% sustained-clock rows do not apply.

### Wall clock

| Stage | Budget | Actual |
|---|---|---|
| Stage 2 — label corpus | 1.25 h | 0.25 h |
| Stage 3 — regime + ranker + calibrator | 0.42 h | 1.25 h |
| tau=1 arm | 0.31 h | 1.13 h |
| Stages 4 → ablations (single driver) | ~12 h | 7.6 h |
| **Full re-run, Stage 2 → ablations** | **15.9 h** | **10 h 18 m** (23:37:35 → 09:56:00) |

Simulation-bound stages ran ~5x **faster** than budgeted; training-bound stages ran ~3x
**slower**. `campaign_budget.py` calibrates against 8 physical cores and a single
throughput constant, so it misestimates in both directions on this machine.

---

## 2. Gates

| Gate | Result |
|---|---|
| `scripts/preflight.py` | PASSED — every module compiles and imports |
| `pytest -q` (pre-clean, artifacts present) | 159 passed, 1 skipped |
| `pytest tests/test_reproducibility.py` | 8 passed |
| `scripts/audit_reviewer_items.py` | **3 FAILURES** — see §5.1 |
| `scripts/clean_stale.py --force` | wiped; kept `S1_calibration`, `S1_perishability`, `figures/S1_calibration` |
| `scripts/campaign_preflight.py` | **exit 0 — READY TO START** |
| Action set | `['EEDD','COVERT','MS','ATC','MDD','EDD']`, ATC k=3.0, COVERT k=4.0 — exact match |

`clean_stale.py` initially **refused** (`data/label_meta.json looks like current-revision
Stage 2`). `--force` was correct here: `simulation/heuristics.py`,
`labeling/rollout_labeler.py` and `experiments/generate_labels.py` all changed in
`045edbc`.

---

## 3. The six watch items

### 1. Label entropy — PASS

Median train row entropy **0.6381**, target band **[0.3870, 0.9048]**,
`entropy_in_band = True`. Beta = 0.469759, `per_row`. No adjustment made.

### 2. `frac_separation_below_1se` — 0.3345, better than feared

**33.4%** at full scale (8000 rows, 250 shifts), against 50.4% on the brief's smoke corpus
and 55.3% on ours. `frac_separation_below_2se` = 0.5604; median separation 1.68; rollout
SE mean 3.71, median 1.81.

The rollout resolves two-thirds of decision epochs. The "half the labels are unresolved"
scenario did not materialise — the smoke corpus was pessimistic. On this number the case
for promoting the M sweep to the headline of §6.4 is weaker than the brief anticipated.
One third unresolved remains a real limitation to state.

### 3. Margin over EEDD-alone — DAHS clears it, but not the way the framing assumes

Paired over 50 shifts, **composite cost**:

| | ours | eedd | diff | 95% CI | ratio | Wilcoxon | wins |
|---|---|---|---|---|---|---|---|
| | 381.42 | 695.77 | 314.35 | **[141.4, 528.8]** | 1.82x | p=1.95e-03 | **21/50** |

Official BH-FDR stats on the **primary metric** (service-failure rate),
`results/E2/default_stats.parquet`: diff **0.0253**, CI **[0.0105, 0.0434]**,
`p_adj_bh = 0.0109`, **`reject_bh = True`**.

`SUBMISSION_CHECKLIST.md` §2 item 1 answers **YES** — the paired interval excludes zero.
§6.2, §7 and the abstract do not need rebuilding.

**Two caveats that must be disclosed:**

- DAHS is strictly cheaper on **21 of 50 shifts**, EEDD on **7**, and they **tie on 22**.
  The mean advantage is tail-driven (EEDD's worst shift is 5281 against DAHS's 2421).
  DAHS also wins the **median** (44.1 vs 57.6). An earlier draft of this report
  wrongly said EEDD won the median.
- **EEDD is not the best single rule.** By aggregate composite cost the best is
  **COVERT (454.36)**; EEDD ranks 8th of 14. Against COVERT, DAHS wins **49/50**, 1.19x,
  p=3.55e-15 — a smaller margin but a far more robust claim. The "EEDD owns 15/16 cells"
  premise is a per-decision oracle statistic and does not survive translation into
  aggregate cost.

### 4. FIFO comparison — 3.90x, NOT the predicted ~1.20x

Composite cost **3.90x** (50/50 shifts, CI [891, 1335]); service-failure rate **2.75x**
(0.0689 vs 0.1894, `reject_bh = True`).

The brief predicted ~1.20x from the *old demo logs* with only the metric rewritten.
That calculation still had the dispatcher idling pickers for arrival-agnostic rules
(F4), which uniquely favoured FIFO. After causal admission, utilisation is 0.956 for
every method including FIFO, and FIFO's failures are late *served* orders
(`sla_breach_rate_arrived` 0.182 vs DAHS 0.065), not abandoned work. The 3.90x is
the post-F4 number; the 1.20x is not a target.

### 5. PPO and offline FQI — the structural claim is withdrawn

`gap_closed_fraction = 0.7831`. The artifact's own verdict:

> Tuning closes a substantial share of the gap. Section 6.9's 'structural, not budgetary'
> claim must be withdrawn and the tuned configuration reported as the PPO baseline.

| factor | spread |
|---|---|
| **normalization** | **313.45** |
| n_steps | 22.48 |
| gamma, gae_lambda, ent_coef | **0.00** |

Best config `norm(obs=True, rew=True)`; baseline 695.77 → best 449.60; DAHS 381.42. Three
of five algorithmic hyperparameters contribute *exactly zero*. The deficit was a missing
observation/reward normalisation wrapper — an implementation detail, not a structural
property. The untuned sweep baseline (695.7707970245864) matches EEDD to 13 decimals: it
collapses to a constant EEDD policy.

**FQI coverage:** effective actions **5.999/6** overall, **5.995/6** breach-prone, 5.937
conditional. Verdict: *"Coverage is adequate: the offline-RL deficit cannot be attributed
to unsupported actions in the hard region."* The artifact also records that under the
**submitted** round-robin logging scheme conditional coverage would be 1.0 by
construction, "contradicting Section 6.10's coverage claim" — the revision's random
behaviour policy is what fixed it.

After the observe-once logger fix DAHS *does* beat FQI, but barely: 381.42 vs 396.80,
1.04x, CI [2.69, 29.72], p=8.05e-03. On the pre-fix tree they were statistically tied.

### 6. Regime K* — selected at the grid endpoint

**K\* = 12**, the maximum of `k_grid = [2,3,4,5,6,7,8,10,12]`. The code emits its own
warning: *"K is being chosen by the grid boundary, not by the data."*

BIC is still falling steeply at the edge — K=8: −168901, K=10: −222525, K=12: **−240139**.
No flattening. `mean ARI 0.9700`, `stable = True`, so the clustering is reproducible; it is
the model order that is undetermined.

`045edbc` added `n_init: 5` (good — removes init luck) but did **not** widen `k_grid`. Not
worked around: `config.yaml` was not edited. The smoke corpus gave an interior K\*=4, so
only full scale exposes this.

---

## 4. An unflagged finding: the rollout horizon buys nothing

Not among the six watch items, but it bears on the method's core design.

**`snapshot_xgb` (the tau=1 arm) is statistically indistinguishable from DAHS (tau=4).**
On the primary metric: ours 0.0689 vs snapshot_xgb 0.0671 — the tau=1 arm is *better* on
the point estimate — diff −0.0018, CI [−0.0042, 0.0005], `p_adj_bh = 0.336`,
**`reject_bh = False`**. On composite cost: 381.42 vs 388.13, CI [−4.05, 17.96] (includes
zero).

`baselines/snapshot_xgb.py` describes this gap as "the headline number".

The tau sweep agrees — `results/E4/tau_summary.parquet`, composite cost:

| tau | point | 95% CI |
|---|---|---|
| 1 | 388.13 | [235.76, 566.33] |
| 2 | **372.31** | [225.11, 545.43] |
| 3 | 373.38 | [225.53, 547.53] |
| 4 | 381.42 | [232.65, 555.07] |

tau=2 is best, tau=4 is third, and every interval overlaps every other. The expensive part
of the method — tau=4, M=20 rollout labelling — is not buying measurable accuracy over
tau=1.

Both MPC teachers still beat the student (greedy_mpc 0.93x, rolling_mpc 0.95x, both CIs
excluding zero), which bounds what §6.11's amortisation argument can claim.

### Table 1 — all methods, composite cost, paired vs DAHS (50 shifts)

| method | composite cost | ratio | 95% CI of diff | wins |
|---|---|---|---|---|
| greedy_mpc | 356.14 | 0.93x | [−39.4, −12.8] | 19/50 |
| rolling_mpc | 362.58 | 0.95x | [−28.1, −10.8] | 15/50 |
| **ours (DAHS)** | **381.42** | — | — | — |
| snapshot_xgb | 388.13 | 1.02x | [−3.6, 18.2] | 32/50 |
| offline_fqi | 396.80 | 1.04x | [2.9, 29.8] | 31/50 |
| covert | 454.36 | 1.19x | [60.9, 85.0] | 49/50 |
| linucb | 551.55 | 1.45x | [94.7, 264.2] | 46/50 |
| atc | 559.92 | 1.47x | [150.9, 206.6] | 50/50 |
| ppo_fair | 610.93 | 1.60x | [114.0, 380.5] | 50/50 |
| eedd | 695.77 | 1.82x | [141.4, 528.8] | 21/50 |
| mdd | 733.15 | 1.92x | [200.9, 540.9] | 49/50 |
| edd | 763.06 | 2.00x | [211.9, 594.5] | 49/50 |
| ms | 789.82 | 2.07x | [225.1, 635.7] | 49/50 |
| wspt | 1215.70 | 3.19x | [753.4, 911.9] | 50/50 |
| fifo | 1485.97 | 3.90x | [895.5, 1342.4] | 50/50 |
| fefo | 1698.96 | 4.45x | [1077.5, 1584.8] | 50/50 |

Paired intervals are `experiments.stats` bootstrap percentile intervals of
(method − DAHS) composite cost, seed 1337, 10,000 resamples (same as
manuscript Table 6). An earlier draft of this table used a different draw
(`[138.50, 524.46]` for EEDD); do not cite that draw.

---

## 5. Section-3 artifacts

Every artifact in RUN_CAMPAIGN.md §3 was produced:

| Artifact | Produced | Key values |
|---|---|---|
| `results/S1_calibration/rule_calibration.json` | yes | ATC k\*=1.5 standalone / 3.0 portfolio; COVERT 4.0 / 4.0 |
| `results/S1_calibration/pool_screening.json` | yes | retained `[EEDD, COVERT, MS, ATC, MDD, EDD]` |
| `results/S1_calibration/diversity_state_grid.parquet` | yes | Stage 1, unchanged |
| `results/S1_perishability/pivotality_summary.json` | yes | Stage 1, unchanged |
| `data/label_meta.json` | yes | beta=0.469759 per_row; entropy 0.6381; sep<1se 0.3345; 4,401,600 interval-steps |
| `runs/phase4/phase4_regime.json` | yes | K\*=12 (grid edge), ARI 0.9700, stable |
| `runs/phase4/phase4_metrics.json` | yes | CV soft-xent 0.87583; ECE 0.1700→0.0213; Brier 0.1730→0.1240 |
| `results/*.parquet` (16 methods) | yes | Table 1 above |
| `results/E2/default_stats.parquet` | yes | bootstrap CI + Wilcoxon + BH-FDR, all four scenarios |
| `results/E3/e3_summary.parquet`, `e3_cost_summary.parquet` | yes | 4 retrain + inference ablations |
| `results/E4/weights_summary.parquet` | yes | objective-weight sweep |
| `results/E4/tau_summary.parquet` | yes | tau in {1,2,3,4}, see §4 |
| `results/E8/robustness_grid_summary.parquet` | yes | 12 untuned cells x 4 methods |
| `results/E10_misspecification/misspecification.parquet` | yes | 5 axes |
| `results/E11_rl_sensitivity/ppo_sensitivity.json` | yes | gap_closed 0.7831 |
| `results/E9/` | yes | FQI eval + data-efficiency + 12-cell grid |
| `runs/data_efficiency/`, `figures/data_efficiency/` | yes | Figure 4, budgets {25,50,100,150,250} x 5 reps |
| `results/A/`, `results/A2/` | yes | Olist validation + arrivals |
| `results/E12_compute/latency.json` | yes | extracted after the run from eval parquets |

**Calibration note:** isotonic calibration improves ECE (0.1700→0.0213, acceptance <0.05
met) and Brier (0.1730→0.1240) but *degrades* soft cross-entropy (0.8280→2.3577).
EDD never wins in the calibration split and is passed through uncalibrated. An
earlier draft of this report quoted the superseded campaign's E5 print
(0.1709→0.0198); the live file is `runs/phase4/phase4_metrics.json`.

### 5.1 `audit_reviewer_items.py` — 3 failures

```
R1.6b RL sensitivity + coverage: PAPER missing 'gap_closed_fraction'
R1.6c composite cost primary:    PAPER missing 'Rank the table by composite cost'
BIB: present but uncited:        ['lundberg2017shap']
```

All three are **paper prose**, introduced by the manuscript rewrite in `045edbc`. None
affects the validity of any measured number. Not fixed here — the brief reserves prose for
the authors.

### 5.2 `scripts/build_submission.py --check` — verbatim

```
[warn ] pandoc not on PATH — conversion unavailable
[warn ] no LaTeX engine on PATH — PDF build unavailable

[gate ] NOT READY — 8 blocking problem(s):
   - 34 unresolved TBD-rerun marker(s). Each states what to report and which way the conclusion falls; resolve against the measured outcome rather than deleting the unfavourable branch.
   - draft scaffolding still present: revision-note blockquote
   - draft scaffolding still present: DRAFT vN comment block
   - figure referenced but missing: ../figures/E4/tau_sla_breach_rate.png
   - figure referenced but missing: ../figures/E8/robustness_grid_heatmap_sla_breach_rate.png
   - in the bibliography but never cited: ['lundberg2017shap']
   - response to reviewers still has unfilled ⟨…⟩ slots
   - cover letter still has unfilled ⟨…⟩ slots

See SUBMISSION_CHECKLIST.md for the order to work through these.
```

**Both "missing" figures were in fact produced**, under the post-rename metric name:
`figures/E4/tau_service_failure_rate.png` and
`figures/E8/robustness_grid_heatmap_service_failure_rate.png`. The manuscript still
references the old `sla_breach_rate` filenames. This is a stale reference in the paper,
not a missing artifact.

---

## 6. Failures and defects

### 6.1 Still open — three sweeps deliberately not run

These exited 2 **by design** ("a printed recipe is not a completed experiment"). Each
needs multi-hour compute and none was run:

| Command | What it needs | Feeds |
|---|---|---|
| `e4_sensitivity n_samples` | 5 labelling passes, M in {1,5,10,20,40} | §6.4 M sweep, TBD line 1821 |
| `e4_sensitivity theta` | 5 labelling passes, multiples {1.5,2,2.2,3,4} | theta sensitivity |
| `e3_ablations relabel single_sample_rollout` | M=1 relabel + retrain | ablation table, TBD line 2042 |

The M sweep matters most: it is the experiment the brief said becomes central if
`frac_separation_below_1se` sits near 50%. At 33.4% it is less critical, but §6.4 still has
no measured M curve.

### 6.2 Resolved during the run

- **Wrong repo.** The initial working directory was a different repository (`Dummy-Repo`)
  with an empty tree and no campaign infrastructure. Cloned the correct repo.
- **Lockfile satisfiable only on Python 3.12.** `pyproject.toml` says `>=3.10,<3.13` but
  `scipy==1.18.0` requires `>=3.12`. The documented "3.10–3.12" range is wrong: only 3.12
  works. Installed 3.12.10. `045edbc` added platform markers for Linux but did **not**
  narrow the documented range — this remains a reproducibility defect for reviewers.
- **`make clean-stale` could not execute.** The Makefile recipe was one physical line with
  literal `\n` sequences instead of continuations. Superseded by `scripts/clean_stale.py`
  in `045edbc`.
- **OneDrive corrupted the wipe.** With the repo on the OneDrive-synced Desktop, `rmtree`
  deleted files but could not remove directories (OneDrive held handles), and
  `ignore_errors=True` swallowed the failures. The empty shells later defeated `cmd_tau`'s
  `run_dir.exists()` guard and crashed the tau sweep. Repo moved to `C:\CAOR`.
- **`make stage4-static` omits four deployed rules** (EEDD, COVERT, MS, MDD), including the
  watch-item-3 comparator. Driven from RUN_CAMPAIGN.md §2 instead. `campaign_preflight`
  validates the doc, not the Makefile — this divergence is unchecked.
- **Stage 4 → Stage 5 ordering inversion.** `e9_offline_fqi robustness_grid` read an E8
  artifact produced a stage later. Masked because `results/E8/` was committed: without
  `clean-stale` it silently compared new FQI numbers against **pre-revision** DAHS numbers.
  Fixed in `045edbc` (`e9_offline_fqi.py:443-452`).
- **`snapshot_xgb` was missing from the Stage 4 method list**, so Figure 4 could not be
  drawn. Fixed in `045edbc`.
- **Orphaned worker pools.** Stopping a driver killed the tracked shell but not the script
  beneath it; its loop respawned 32-worker loky pools. 36 stray processes were traced and
  killed. Anyone stopping these runs should kill the `bash.exe` parent, not the children.

---

## 7. TBD-rerun passages with no number from this campaign

34 `⟨TBD-rerun⟩` markers remain in `paper/manuscript.md`. The campaign produced numbers for
most. These are the ones it **did not**:

| Line | Passage | Why |
|---|---|---|
| 1821 | companion sweep over number of continuations (M) | `e4_sensitivity n_samples` not run (§6.1) |
| 2042 | both ablations incl. single-sample rollout | `e3_ablations relabel` not run; same compute as M=1 |
| 2286 | per-decision latency | **now extracted** to `results/E12_compute/latency.json` from the eval parquets |
| 2275 | measured interval-steps / wall-clock | **partial:** `data/label_meta.json` + `results/E12_compute/label_budget.json`. Still missing `python -m experiments.compute_budget measure` |
| 2306 | sample-efficiency curve at pool sizes 2, 4, 8 | no retrain-at-pool-size driver. `compute_budget scaling` is a different experiment (successive-halving vs uniform on the *screening* pool) |
| 2320 | arg-max agreement and label KL vs uniform | produced by `python -m experiments.compute_budget scaling`, not yet run |

The remaining markers have their numbers in the artifacts listed in §5, but the prose is
the authors' to write.

---

## 8. Summary judgement

On the **default** scenario the core claim survives: **DAHS beats every static dispatch
rule** on composite cost and service-failure rate, with paired intervals excluding zero
after BH-FDR. It does **not** beat the teachers, is tied with tau=1, loses to WSPT under
`high_load_perish`, and loses to EEDD (by pennies) under `balanced`.

Three things the campaign changed:

1. **Two structural claims retire.** §6.9's PPO argument is withdrawn by the code's own
   verdict (78.3% of the gap closes with a normalisation wrapper). §6.10's coverage
   explanation is ruled out by the coverage diagnostic.
2. **The rollout horizon is not earning its cost.** tau=1 is statistically
   indistinguishable from tau=4, and the tau sweep's intervals all overlap (§4). This is
   the most consequential unflagged finding.
3. **The EEDD margin is tail-driven** — 21/50 shifts — and EEDD is not the strongest single
   rule. COVERT is, and the DAHS-over-COVERT result (49/50, 1.19x) is the more defensible
   headline.

K\* at the grid boundary (§3.6) still needs an author decision: report it as a
limitation, or widen `k_grid` and retrain (that invalidates Stage 3 onward). The FIFO
3.90x is explained in §3.4 — do not treat 1.20x as a target.

---

## Remaining-run addendum

Executed against `f1ee3ff` per `RUN_PROMPT.md`. Sections **A (must)** and **B (should)**
were run. Section **C (theta)** was **not run** — the prompt marks it optional and it was
not requested.

### Gate

`preflight.py` passed. `pytest -q` gave **1 expected skip** (FEFO-not-in-pool) and **one
failure**, which is a false positive and is described in §6.3 below. Pool check printed
exactly `['EEDD', 'COVERT', 'MS', 'ATC', 'MDD', 'EDD'] 3.0 4.0`. `C:\CAOR` confirmed off
OneDrive.

### A — M-sweep (`e4_sensitivity n_samples`)

Wall clock **2 h 41 m** total (10:52:53 → 13:34:28). Training dominates and is flat in M:
the HP search costs ~35 min per arm regardless of M, so only M=40 is meaningfully more
expensive.

| M | labelling | training | arm total | median entropy | in band? |
|---|---|---|---|---|---|
| 1 | 49 s | 34.6 min | 35.7 min | 0.6413 | yes |
| 5 | 3.9 min | ~28 min | 32.6 min | 0.6441 | yes |
| 10 | ~7 min | ~28 min | 35.3 min | 0.6493 | yes |
| 40 | ~28 min | ~30 min | 58.1 min | 0.6520 | yes |

Target band throughout: **[0.3870, 0.9048]**. Every pass printed `-> OK`;
`--force-out-of-band` was never used.

#### Result — the multi-sample estimator buys nothing

| M | composite cost | service-failure rate | mean tardiness | paired vs M=20 |
|---|---|---|---|---|
| **1** | **380.24** | 0.0685 | 0.7688 | −1.18, p=0.910 |
| 5 | 380.44 | 0.0680 | 0.7650 | −0.98, p=0.539 |
| 10 | 382.40 | 0.0691 | 0.7786 | +0.98, p=0.125 |
| 20 | 381.42 | 0.0689 | 0.7753 | committed main run |
| 40 | 382.78 | 0.0686 | 0.7668 | +1.36, p=0.352 |

The curve spans **2.5 units on a base of 381 (0.7%)**, every bootstrap interval overlaps
every other, no paired test approaches significance, and **M=1 is nominally the best cell
on composite cost**.

This is watch item 2 of the remaining-run prompt resolving in the direction it warned of:
*"if M=1 matches M=20, the multi-sample machinery bought nothing and that is the result."*

Read together with §4 of the main report — `snapshot_xgb` (tau=1) statistically
indistinguishable from DAHS (tau=4) — **both axes of the rollout estimator are flat**. The
labelling scheme costs |H|·tau·M = 480 simulated interval-steps per decision and performs
no better than the |H|·1·1 = 6-step corner. Neither the horizon nor the number of
continuations is earning its compute.

#### E3 single-sample ablation

`results/E3/single_sample_rollout.parquet` is present (copied from
`results/E4/n_samples/M_1/ours.parquet` as the prompt directs) and `e3_summary` was
**regenerated** — but only after working around a bug, see §6.4.

Regenerated E3 table, composite cost, vs `ours` 381.4196:

| ablation | composite cost | diff | p_raw | p_adj_bh | reject_bh |
|---|---|---|---|---|---|
| no_switching_controller | 380.4286 | −0.9911 | 0.5940 | 0.8315 | False |
| **single_sample_rollout** | **380.2351** | **−1.1846** | 0.9095 | 1.0000 | False |
| no_calibration | 381.4599 | +0.0403 | 0.5699 | 0.8315 | False |
| random_ambiguity_filter | 381.4196 | 0.0000 | 1.0000 | 1.0000 | False |
| hard_labels | 382.4306 | +1.0110 | 0.4651 | 0.8315 | False |
| top5_features | 383.1432 | +1.7235 | 0.4204 | 0.8315 | False |
| no_regime | 383.5620 | +2.1424 | 0.0479 | 0.3355 | False |

**No ablation is significant after BH-FDR.** `no_regime` is closest and still does not
survive correction. Removing the regime layer, using hard labels, dropping to five
features, and disabling the switching controller all fail to move composite cost
measurably.

### B — compute budget

Wall clock **~11 min** (13:35:54 → 13:47:04). `analytic` instant, `measure` 1.9 min,
`scaling` 9.3 min. `latency.json` and `label_budget.json` were left untouched as
instructed.

`results/E12_compute/measured_throughput.json`:

| | |
|---|---|
| machine | Intel Core i9-14900K, 24c/32t |
| single-core throughput | **389.3 interval-steps/s** |
| seconds per shift | 37.69 |
| full corpus (3,848,000 steps) | **2.75 h single-core** |

`analytic`: revision total 3,848,000 interval-steps at M=20 — **6.17x** the submitted
cost for **20x** the samples per cell, because the O(N²) replay term is gone.

`results/E12_compute/successive_halving.json` — **a negative result**:

| | |
|---|---|
| argmax agreement rate | 0.85625 |
| mean label KL | 0.1851 |
| steps, uniform | 109,800 |
| steps, successive halving | 108,580 |
| step saving | **1.1%** |
| verdict | *"Successive halving degrades the label at this budget; report uniform allocation and note the mitigation as unsuccessful here."* |

Successive halving buys a 1.1% compute saving while disagreeing with the uniform-allocation
arg-max on 14% of epochs. It should be reported as an unsuccessful mitigation.

`pool_scaling_analytic.parquet` covers pool sizes 2/4/8/10/16/20 — steps scale linearly
(0.50x / 1.00x / 2.00x relative to a 4-rule pool).

### Assembled artifacts

- `results/E4/n_samples_summary.parquet` — schema **verified identical** to
  `results/E4/tau_summary.parquet`: `knob, value, metric, point, ci_lo, ci_hi, n,
  n_resamples`. Metrics: `service_failure_rate`, `composite_cost`, `mean_tardiness`.
  M=20 cell taken from the committed `results/ours.parquet`.
- `figures/E4/n_samples_composite_cost.{png,pdf}`
- `figures/E4/n_samples_service_failure_rate.{png,pdf}`

Built by importing `_bootstrap_summary` and `_plot_curve` from
`experiments.e4_sensitivity`, so the schema and plot style match the tau sweep exactly
rather than being reimplemented.

### Effect on §7 of the main report

Three of the five gaps listed there are now closed, and one was mislabelled:

| Line | Status after this run |
|---|---|
| 1821 — M sweep | **closed** — `results/E4/n_samples_summary.parquet` |
| 2042 — single-sample ablation | **closed** — in `results/E3`, summary regenerated |
| 2286 — per-decision latency | **already closed** — `results/E12_compute/latency.json`, committed in `f1ee3ff` |
| 2320 — arg-max agreement and label KL | **closed** — `successive_halving.json` (0.85625 / 0.1851). The main report wrongly said no producer existed; `compute_budget scaling` is the producer. |
| 2306 — sample-efficiency curve at pool sizes 2/4/8 | **still open** — `pool_scaling_analytic.parquet` gives the *analytic cost* at those pool sizes, not an empirical sample-efficiency curve. Producing one needs a re-label and retrain per pool size; no driver exists. |

### Anything that failed

**§6.3 — `test_campaign_commands.py` false positive (not fixed; prose).**
`pytest -q` fails one case:
`test_campaign_command_parses[experiments.compute_budget measure` and `scaling` were also never]`.
The extractor matches `python -m …` inside backticks but continues past the closing
backtick. `RUN_CAMPAIGN.md:311` reads:

> `` `python -m experiments.compute_budget measure` and `scaling` were also never ``

so it executed `compute_budget measure and scaling were also never --help`. All three
subcommands (`analytic`, `measure`, `scaling`) were verified to parse cleanly and all three
ran to rc=0 in section B. Fix is either to reword that sentence or to stop the extractor at
the closing backtick. Not fixed here — it is prose.

**§6.4 — `e3_ablations summary` is not idempotent (worked around; no code changed).**
`cmd_summary` globs `results/E3/*.parquet` and skips only the stems `ours` and
`e3_summary`, but **not `e3_cost_summary`** — its own second output. The first run succeeds;
every subsequent run re-ingests its own cost table and dies:

```
KeyError: "KPI columns ['service_failure_rate', 'composite_cost'] are not in this result
frame. Available: ['ablation', 'decision_latency_ms_mean', 'decision_latency_ms_p95',
'eval_wall_clock_s_per_shift', 'retrained', 'train_flags', 'train_wall_clock_s']"
```

Worked around by moving `e3_cost_summary.parquet` aside and re-running; the command then
exited 0 and regenerated both outputs. One-line fix: add `e3_cost_summary` to the skip
tuple at `experiments/e3_ablations.py:467`.

**§6.5 — the `random_ambiguity_filter` ablation cannot affect KPIs.**
Its row reads `diff = 0.0000, p = 1.0000`. Every KPI column in
`results/E3/random_ambiguity_filter.parquet` is **bit-identical** to `results/ours.parquet`
(max absolute difference exactly 0.0 on `composite_cost`); only `wall_clock_s`,
`decision_latency_ms_mean` and `decision_latency_ms_p95` differ, and those are measurement
noise.

The module docstring explains why: the ablation drops "the same NUMBER of **test rows** at
random as the theta filter drops by confidence". `evaluate --method ours` runs the policy in
simulation over 50 shifts and never consults the filtered test-row set, so the KPI parquet
is identical **by construction**. The retrain is real (1623.8 s) but trains on an unchanged
training set.

That row is an identity, not a measurement. Reporting it as "the ambiguity filter has no
significant effect on cost" would be misleading — its effect can only appear in label-space
metrics (arg-max agreement, soft cross-entropy on retained rows), not in KPIs.

### Not run

**Section C — theta sweep.** Optional per the prompt. On the training times measured here
(~35 min per arm, flat in the knob) four arms would cost roughly **4 h**, less than the
6 h the prompt budgets. Nothing else in the report depends on it.
