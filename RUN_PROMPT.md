# The prompt for the campaign machine

Paste everything between the `=====` lines into the agent on the machine that
will run the campaign, after cloning the repo and `git pull`-ing to at least the
commit that added this file.

Budget roughly **6 hours on 16 cores**. Stage 1 is already done and must not be
re-run; see `RUN_CAMPAIGN.md`.

---

```text
=====================================================================
You are running the CAOR-D-26-01812 revision campaign for the DAHS project. The
repository is cloned and is the current working directory. Your job is to execute
the compute campaign and commit the results. Do not redesign anything, do not
"improve" the method, and do not tune anything to make a number look better.

READ FIRST
  RUN_CAMPAIGN.md  — the stage list, costs, and what to extract
  REVISION_PLAN.md — why each change exists, mapped to reviewer comments
Both are authoritative. This prompt is the operating procedure, they are the spec.

SETUP
- Python 3.10-3.12 only (3.13+ will not work).
- Install from the LOCKFILE, not from the pyproject ranges. The paper claims the
  pipeline is bit-reproducible and the ranges do not deliver that across machines:
      python -m venv .venv
      .venv/bin/pip install -r requirements-lock.txt      # Windows: .venv\Scripts\pip
      .venv/bin/pip install -e . --no-deps
- If a locked version will not install on this platform, STOP and report which
  one. Do not substitute a different version silently — that breaks the
  cross-machine comparability the lockfile exists for.

GATE — do not skip, do not proceed past a failure
      .venv/bin/python scripts/preflight.py            # ~2s, imports every module
      .venv/bin/python -m pytest -q                    # must exit 0
      .venv/bin/python scripts/audit_reviewer_items.py # must print ALL CHECKS PASS
Expect ~85 passed and ~11 skipped. Every skip should name a campaign artifact that
does not exist yet, or say "FEFO is not in the deployed pool". Any OTHER skip, or
any failure, means stop and report.

WIPE THE PRE-REVISION ARTIFACTS
      make clean-stale
This is mandatory, not tidiness. The committed data/, runs/ and results/ predate
the corrected objective, and inserting the calibration seed block shifted the test
seeds so old and new results overlap on only 20 of 50 seeds — a paired comparison
across them is silently misaligned rather than obviously empty.

`clean-stale` deliberately KEEPS results/S1_calibration, results/S1_perishability
and figures/S1_calibration. Those are current-revision Stage-1 results, not stale
ones; config.yaml's fitted ATC/COVERT scales and its screened pool are derived
from them and Section 6.1 of the manuscript reports them. After running it,
confirm they are still there. If they are gone, restore with
`git checkout -- results/S1_calibration results/S1_perishability config.yaml`.

DO NOT RE-RUN STAGE 1. It is complete. Verify before starting Stage 2:
      .venv/bin/python -c "from omegaconf import OmegaConf; c=OmegaConf.load('config.yaml'); print(list(c.heuristics.pool), c.heuristics.atc_lookahead_k, c.heuristics.covert_lookahead_k)"
Must print exactly:
      ['EEDD', 'COVERT', 'MS', 'ATC', 'MDD', 'EDD'] 3.0 4.0
If it prints anything else — in particular if the pool still contains FIFO, WSPT
or FEFO — run `.venv/bin/python scripts/apply_stage1.py` and check again. Labelling
the wrong action set wastes the whole campaign.

SMOKE BEFORE THE EXPENSIVE STAGES
      make stage2-smoke      # 3 train + 2 test shifts, ~7 min
      make stage3-smoke
stage2-smoke must print "median train row entropy = ... -> OK". If it prints OUT
OF BAND, stop and report the number — do not proceed and do not adjust the
temperature to force it in.

THE CAMPAIGN — run the stages in RUN_CAMPAIGN.md section 2, in order. Commit and
push after each stage so a crash at hour four does not cost hours one to three.
Stage 2 must precede Stage 3, and `make tau1` must complete before Stage 5.

WHAT TO WATCH, AND REPORT RATHER THAN FIX
These are known risks. Each is a finding to report, not a bug to work around. If
you find yourself changing a parameter to make one of them go away, stop.

1. Label entropy band. Stage 2 prints the achieved median against the target.
   Report it either way.
2. frac_separation_below_1se in data/label_meta.json — the share of decision
   epochs whose best and second-best rules are separated by less than one pooled
   standard error. On the smoke corpus this was 76.8%. If it stays that high at
   full scale, the soft labels are largely noise; that is a real result about the
   method and it must be reported, not smoothed. It also makes the M sweep
   (E4 n_samples) the important experiment rather than a supplementary one.
3. The margin over EEDD-alone. EEDD wins 65% of decisions and owns 15 of the 16
   state-space grid cells; the per-cell oracle gap is only 7.29 points. Watch
   DAHS's COMPOSITE COST margin over the eedd row in Stage 4. If it is not
   statistically meaningful, that is the headline finding and Section 6.2 must be
   written around it.
4. The FIFO comparison. Counting unserved-and-overdue orders as failures moved the
   advantage over FIFO from ~3.8x to ~1.20x on this repo's own committed logs.
   Expect roughly that, not the submitted margin.
5. PPO and offline-FQI. Sections 6.9 and 6.10 are written CONDITIONALLY on the
   sensitivity sweep and the coverage fix. If tuning closes a material part of
   either gap, the structural claim is withdrawn — that is the honest outcome and
   the sections already say so. Report gap_closed_fraction.
6. Regime K*. If the BIC sweep selects at an endpoint of K in {2..12}, report it:
   that means the grid chose K, not the data.

DO NOT
- Do not parallelise LinUCB. It is an online learner whose weights persist across
  shifts by design; parallel_safe = False enforces this and the harness honours it
  over any --n-jobs.
- Do not edit config.yaml except via scripts/apply_stage1.py.
- Do not re-run a stage with different settings to get a different answer.
- Do not fill in any TBD-rerun passage in paper/manuscript.md yourself. Collect the
  numbers; the authors write the prose.

FINAL REPORT
Produce a markdown file CAMPAIGN_REPORT.md at the repo root containing:
- Machine: CPU model, core count, RAM, OS, Python version. Total wall-clock per
  stage.
- The gate results (preflight, pytest counts, audit).
- For every artifact in RUN_CAMPAIGN.md section 3 ("Numbers to extract"): the file,
  whether it was produced, and the key values in it.
- The six watch items above, each with its measured value and a one-line reading.
- Anything that failed, what you did, and whether it is resolved or still open.
- An explicit list of any TBD-rerun passage in paper/manuscript.md that the
  campaign did NOT produce a number for, so the gaps are visible rather than
  discovered at submission.

Commit CAMPAIGN_REPORT.md with the results and push.
=====================================================================
```

---

## Why Stage 1 is excluded

Stage 1 ran at commit `b76b6b4` on the 30-shift calibration block, under the
corrected objective and dispatcher. Since then the only changes to anything it
depends on are `labeling/soft_label_converter.py` and the `labeling.beta_*` keys
in `config.yaml` — and Stage 1 does not import the soft-label converter.
`simulation/`, `regime/`, `models/`, `experiments/calibrate_rules.py` and
`experiments/perishability_diagnostic.py` are byte-identical to when it ran.

Re-running it would cost about 35 minutes (2.97M interval-steps, almost all of it
the ATC/COVERT k-sweep) and produce the same numbers. The one thing that *was*
missing — `apply_stage1.py` after the `screen` step, which is what writes the
retained pool back into `config.yaml` — has been applied and committed.

## Why everything else must re-run

| Change | Landed in | Invalidates |
|---|---|---|
| Unserved-and-overdue orders charged; service-failure rate primary | `b9ae082` | every KPI and every cost |
| Two deadline clocks, spoilage priced, priority weights applied | `b9ae082` | the objective, so every label |
| Causal admission — no picker reserved for future arrivals | `b9ae082` | every trajectory |
| Labels are M=20 Monte Carlo means, not one replayed path | `b9ae082` | every training target |
| Calibration seed block inserted | `b9ae082` | the test seeds — old and new overlap on 20 of 50 |
| Feature set 25 -> 26; two degenerate features dropped | `b9ae082` | the model input and the regime layer |
| Per-row label temperature | `133ff02` | every label again |
| Pool 9 candidates -> 6 retained, EEDD added | `0600d98` | the action set, so labels and every baseline |

Nothing downstream of Stage 1 survives any one of these, let alone all eight.
