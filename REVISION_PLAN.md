# CAOR-D-26-01812 — Revision Plan

Response to Reviewers 1, 2, 3, 5. Written against commit `9608465`.

---

## 0. The structural finding

Four reviewer demands each independently invalidate every number in the paper:

| Demand | Reviewer | What it changes |
|---|---|---|
| Count unserved-overdue orders as breaches | R2.1 | the primary metric |
| Average rollouts over multiple continuations | R2.3 | every training label |
| Calibrate ATC; expand and screen the rule pool | R1.4c, R1.4d | the action set |
| Make perishability real (expiry ≠ due date; spoilage in objective) | R1.1c/d/f, R2.2 | the problem definition |

Patching these one at a time means four full re-runs. **Land all model changes first, then re-run once.**

I also found two simulator defects that *cause* symptoms reviewers flag but none of them
diagnosed (F1 and F4 below). If we re-run without fixing them, the revision reproduces the
same "counterintuitive results" (R1.6a) and the same under-powered baselines (R1.6b).

---

## 1. Simulator and model corrections — must land before any re-run

### F1. The objective ignores the priority weights the rules optimise → answers R1.6a, R1.1c

`labeling/snapshot_labeler.py:28` — the cost is

```
J = 3.0·n_breach + 0.2·Σ tardiness + 0.005·|Q|
```

Every order counts 1. But `simulation/heuristics.py:37,68` — WSPT and ATC both rank by
`PRIORITY_WEIGHTS[priority_class] / p`, weights `{low:1, medium:2, high:4}`, class mix
`[0.5, 0.35, 0.15]`. **WSPT and ATC optimise a weighted objective the evaluation never
measures.** WSPT looks bad in Table 1 partly because it is graded on a different exam.

This is the single cleanest answer to R1.6a ("how does WSPT perform worst in throughput?
It should particularly succeed in that metric").

**Fix.** Make the objective weighted and consistent with the rules:

```
J = Σ_{o served}  w_o·[W_b·1(f_o > d_o) + W_t·max(f_o − d_o, 0)]
  + Σ_{o unserved} w_o· W_b·1(d_o < T_end)          # see F3
  + W_s·Σ_o w_o·1(spoiled)                          # see F2
```

Alternative: delete priority classes entirely and use unweighted WSPT/ATC. Cheaper, but
throws away a state feature and a modelling dimension. Recommend the weighted objective.

**Files:** `labeling/snapshot_labeler.py`, `experiments/evaluate.py:74`, `simulation/kpis.py`,
`config.yaml`.

---

### F2. Perishability is decorative → answers R1.1c, R1.1d, R1.1f, R2.2

Facts from the code:

- `is_perishable` appears in exactly four places: order generation, the `pct_perishable`
  feature, the `spoilage_rate` KPI, and the FEFO mask threshold. **It never enters the cost.**
- There is no expiry timestamp. `simulation/kpis.py:40` defines spoilage as
  `perishable AND finish_time > sla_due` — i.e. spoilage *is* an SLA breach, by definition.
- `simulation/heuristics.py:31` — `fefo()` sorts by `o.sla_due`. **The rule labelled FEFO is
  EDD (earliest due date).** R1.1f is exactly correct, and this is a naming error in the code,
  not only in the prose.

**Fix — Option A ("make it real"), recommended:**

1. `Order` gains `expiry_time: float | None`. Perishables draw a shelf life independent of
   `sla_due`; non-perishables get `None`.
2. True `FEFO` sorts perishables by `expiry_time`, non-perishables last.
3. Add `EDD` as its own rule sorting by `sla_due` — this is what the current "FEFO" actually
   was, so existing intuition transfers cleanly and the rename is traceable.
4. Spoilage = `finish_time > expiry_time`, or never served with `expiry_time < T_end`.
   Enters the objective with weight `W_s`.
5. New diagnostic answering R1.1d *quantitatively*: the fraction of decision epochs at which
   delaying an order by one 15-minute interval flips it from in-date to expired. If that
   fraction is negligible, perishability does not matter at this horizon and we say so and
   drop the claim — which is the honest outcome R1.1d is fishing for.

**Fix — Option B ("drop it"):** remove perishability, retitle to *deadline-constrained*,
rename FEFO→EDD, delete the FEFO mask and the spoilage KPI. Smaller paper, fully defensible,
much less compute.

**Files:** `simulation/orders.py`, `simulation/heuristics.py`, `simulation/warehouse_env.py`,
`simulation/kpis.py`, `labeling/soft_label_converter.py` (mask), `models/switching_controller.py`
(mask), `config.yaml`.

---

### F3. The breach rate exempts orders that are never served → answers R2.1

`simulation/kpis.py:37` — `sla_breach_rate = breaches / len(completed)`. Orders still queued
at shift end appear in neither numerator nor denominator; they land in a separate `unfinished`
field that no table in the paper reports. In the cost, `W_u = 0.005` against `W_b = 3.0` — a
**600× discount for never touching an order**. R2.1's arithmetic is right.

**This is not theoretical.** Measured directly from the committed demo run logs
(`demo/dahs-app/runs/`, 10 seeds, frozen model, default config, exact order-level counts):

| | SLA-breach rate (paper's metric) | % of *arrived* orders not shipped on time |
|---|---:|---:|
| DAHS | **3.10%** | **15.00%** |
| FIFO | 11.75% | 17.97% |
| ppo_fair | 9.40% | 16.44% |

The advantage over FIFO goes from **3.8× to 1.20×**; over PPO from **3.0× to 1.10×**. On
several individual seeds it inverts (seed 42: DAHS 4.02% breach vs PPO 4.17%, but 17.41%
unserved vs 12.10%). Applying the same arithmetic to Table 1 itself
(arrivals ≈ 1.65 × 480 = 792/shift; unfinished ≈ 792 − throughput) gives DAHS ≈ 10.1%,
ppo_fair ≈ 10.0%, FIFO ≈ 11.4%.

**We must know this before choosing how to frame the revision.** It does not sink the paper —
the *relative* training-signal comparison survives — but the headline margin does not.

**Fix.**

1. New primary metric `service_failure_rate` = (late-served + unserved-and-overdue) / arrived.
2. Keep `sla_breach_rate` alongside, with its formula stated explicitly in the text (R2.1
   asks for this in so many words).
3. Price unserved orders correctly: an unserved order whose due time has passed simply *is* a
   breach, at weight `W_b`, not `W_u = 0.005`.
4. Report `arrived`, `served`, `unserved`, `dropped` as columns in Table 1.

**Files:** `simulation/kpis.py`, `labeling/snapshot_labeler.py`, `experiments/evaluate.py`.

---

### F4. The dispatcher reserves pickers for orders that have not arrived → answers R1.6a

`simulation/warehouse_env.py:176-202`:

- orders are admitted with `arrival_time <= interval_end` — **15 minutes of look-ahead**;
- then `start = max(picker_free, o.arrival_time, interval_start)`.

So ranking a not-yet-arrived order **reserves a picker and idles it until that order arrives**.
Arrival-sorted rules (FIFO) never pay this. Arrival-agnostic rules (WSPT, ATC) pay it
constantly. Table 1: FIFO utilisation 0.983, WSPT 0.686 — with a queue sitting near the 200
cap. A picker cannot be 31% idle with ~180 orders waiting unless the dispatch model idles it.

Two further consequences: the state vector sees up to 15 minutes of future arrivals (undisclosed
clairvoyance), and the `break` on line 198 stops the whole dispatch loop rather than skipping
one infeasible candidate.

**Fix.** Admit only `arrival_time <= interval_start`; `continue` instead of `break`. This makes
the dispatcher causal and removes a structural handicap that has nothing to do with the rules'
merits.

---

### F5. Constant and collinear state features corrupt the regime layer → answers R1.3a

- `time_to_next_expected_carrier = 1/arrival_rate` (`state_extractor.py:129`) is **constant**
  within a configuration — zero variance, zero information.
- `interval_index_in_shift + intervals_remaining = 32` **exactly** — a perfect linear dependence.

`regime/regime_discovery.py:81-91` fits `covariance_type="full"`, `reg_covar=1e-6`, `n_init=1`
on these raw columns. The covariance is singular; only `reg_covar` prevents a crash, and each
extra component buys likelihood by collapsing further onto the degenerate directions. That is
why BIC falls monotonically to the edge of the grid (K=3 −242,799 → K=6 −287,883, with the 5→6
step **8× larger** than 4→5) and why K=6 is "selected" at the boundary. The reported ARI of
0.998 says EM finds the same degenerate basin every time, not that the structure is real.

**Fix.** Drop both features; re-run the K sweep on a grid that can actually turn (K ∈ 2…12) and
report the curve; add `n_init ≥ 5`. Also run the `no_regime` ablation, which is declared in
`config.yaml` but was **never executed** — the regime layer is a named method component (§4.5)
and the paper never shows what removing it costs.

---

## 2. Method changes the reviewers demand

### M1. Multi-sample rollouts (R2.3) — and it comes out cheaper

`labeling/snapshot_labeler.py:85-90` runs **one** `fast_forward` (full replay from t=0) per
heuristic, and the "future" is the pre-sampled realisation from the shift seed. So each label
is hindsight-optimal for a single path, rollout variance is identically zero, and §4.4's
bias–variance argument has no variance term in the code. R2.3 is correct.

**Fix.**

- Add `WarehouseEnv.branch(rollout_seed)`: deep-copy the state at `t`, discard the
  not-yet-arrived tail of `_all_orders`, resample it from `np.random.default_rng(rollout_seed)`
  with `rollout_seed` derived deterministically from `(shift_seed, t, m)`. Genuine Monte Carlo,
  still bit-reproducible.
- Label with `Ĵ_h(s_t) = (1/M) Σ_m Ĵ^τ_{h,m}` and **record the per-cell standard error** —
  R2.3 asks for the variance explicitly.
- Restate Proposition 1 with truncation bias and estimator variance separated, adding the
  O(1/√M) term. This also repairs §6.4, whose "variance" explanation currently describes
  something the code does not do.

**Compute: this gets faster, not slower.** Today labelling is
O(n_shifts · n_intervals² · n_rules) because every snapshot replays from zero, four times.
Walking `t` forward once per shift and deep-copying at each boundary makes it
O(n_shifts · n_intervals · n_rules · M) — a ~32× cut on the replay term. At M=20 that is still
less total work than today. And the paper's own sample-efficiency result licenses 50 training
shifts instead of 250, which pays for M=20 outright.

---

### M2. The online τ=4 rollout MPC baseline (R2.6) — the missing teacher

`baselines/greedy_mpc.py` is τ=1. DAHS distils a τ=4 rollout. **The thing DAHS claims to
amortise is never evaluated.** Without it we cannot say whether DAHS approaches, matches, or
exceeds its teacher — which is the whole amortisation argument.

**Fix.** `baselines/rolling_horizon_mpc.py`: at each epoch evaluate all rules over τ intervals
averaged over `n_samples` branches, commit the best for one interval, replan. Report KPIs **and
per-decision wall-clock**. This simultaneously answers R3.1 and R3.5 on computational cost.

---

### M3. Calibrate ATC; expand and screen the pool (R1.4a–e)

R1.4c is exactly right and the code confirms it. In `atc()`, `exp(−slack/(k·p̄)) → 1` as
k → ∞, leaving the WSPT key `w/p` — so **WSPT is the k→∞ limit of ATC** and a properly
calibrated ATC cannot lose to it. `config.yaml` sets `atc_lookahead_k: 2.0` and **no search
over k exists anywhere in the repository.**

**Fix.** New `experiments/calibrate_rules.py` performing both calibrations the reviewer asks
for, on a calibration seed set disjoint from train and test:

- (i) **standalone** k* minimising composite cost with ATC used alone (needed because ATC is a
  benchmark);
- (ii) **portfolio** k* maximising realised cost when ATC sits in the DAHS pool.

Report both and the deployed value.

**Pool expansion (R1.4d).** Add EDD, MDD, MS (minimum slack), S/RPT, COVERT, ATCS, CR, SPT,
LPT, and true FEFO; screen with the existing pilot gate (win rate ≥ 8%, top ≤ 60%) and **report
the screening table** — the reviewer explicitly allows "a screening process then eliminates
most of them."

**Complementarity across the state space, not across shifts (R1.4e).** Replace the per-shift
heatmap (Figure 1) with win-rate over a (queue-length × deadline-pressure) grid, plus a
marginal-contribution analysis showing each retained rule earns its slot.

**Answer R1.4b honestly.** On the deployed τ=4 labels, FIFO is the arg-min in **0 of 865**
filtered test states and ATC in **6** (`runs/phase4/phase4_metrics.json`,
`argmax_distribution_test_truth`). Either FIFO earns its place under the corrected model, or we
drop it and say why. Note this also undercuts §6.1's "pool is genuinely diverse" claim, which
is computed from the *pilot* cost data, not from the labels that train the deployed model.

---

### M4. RL baselines: run the sensitivity, then interpret (R1.6b)

`baselines/ppo_fair.py:165-176` uses stock SB3 hyperparameters, **no `VecNormalize`**, no
reward scaling, on raw observations spanning 0–200 (queue length) to 0–1 (utilisation) — and no
tuning at all, while DAHS got an 18-config grid and FQI got 12. §6.9's claim that "the issue is
structural, not budgetary" is unsupported: only the *budget* was varied (8k vs 500k).
`baselines/linucb.py` has the same problem — ridge with `A = I` on unstandardised features is
dominated by whichever column has the largest scale.

**Fix.** Run exactly the sweep R1.6b names — discount γ, GAE λ, rollout length `n_steps`,
entropy coefficient, and observation/reward normalisation on/off — report the full grid, and
only then interpret. Standardise LinUCB's features. Add the offline action-coverage analysis
for FQI **restricted to breach-prone states** (visitation counts under the round-robin
behaviour policy in the high-|Q| / low-slack region), which is the second half of R1.6b.

Also note for fairness: `offline_fqi` sees only the 25-D state while DAHS sees 31 (with regime
posteriors), and deploys as a bare `argmax Q` with no calibration wrapper — so the claim that
the comparison "isolates the training signal" is stronger than the setup supports. Equalise or
qualify.

---

### M5. Model misspecification (R2.5)

Proposition 1 bounds truncation error, not model error. Add an explicit
**label-simulator ≠ evaluation-simulator** experiment: label under nominal parameters, evaluate
under perturbed arrival rate, processing-time distribution, due-window, and picker count;
report degradation against perturbation magnitude. The existing E8 grid and the Olist-arrival
test are close, but are framed as "untuned configurations" rather than misspecification.
Reframing plus a proper perturbation sweep answers R2.5 directly.

---

### M6. Feature provenance, correlation, parsimony (R1.3a, R3.4)

- A table mapping each of the 25 features to its literature source or stated design rationale.
- Correlation matrix and VIF; drop the constants and exact dependencies from F5.
- The top-5-feature ablation R3.4 asks for, plus a feature-count sweep.
- Answer R1.3b in one line: the 1600 test states are 50 held-out shifts × 32 intervals, filtered
  to 865 by the θ=0.55 ambiguity gate.

---

## 3. Text — three new or rewritten sections

### T1. New Section 3: the sequential decision model (R5.1, R5.3, R2.4, R1.1a)

Written in Powell's canonical framework, which is what R5 asks for and which simultaneously
answers R2.4:

- **State** `S_t` = (queue with per-order attributes, picker-availability vector, clock) — the
  *true* state.
- **Decision** `x_t` = a rule from pool `H`.
- **Exogenous information** `W_{t+1}` = arrivals and their attributes in `(t, t+1]`.
- **Transition** `S_{t+1} = S^M(S_t, x_t, W_{t+1})` — the dispatch model, written out explicitly
  (this is also where F4's admission rule gets stated).
- **Objective** `min_π E Σ_t C(S_t, X^π_t(S_t))`.
- **Policy class**: `X^π(S_t) = argmax_h f_θ(φ(S_t))_h` — a *policy-function approximation* over
  a feature map `φ`.

Then state plainly what R2.4 demands: `x_t = φ(S_t)` is an **observation, not a sufficient
statistic**; the problem is a POMDP and DAHS is a PFA on a hand-crafted belief summary. Give
the two-queues-same-φ counterexample, and quantify the approximation cost (label disagreement
among states with near-identical φ).

This one section resolves R5.1, R5.3 (model), R5.4 (density), R2.4, and much of R1.1a.

### T2. Rewritten Section 2 and repositioning (R1.2a–e, R5.2, R5.3)

The novelty claim has to be retired. What DAHS does — simulate each action at sampled states,
train a classifier on the outcome — is:

- **Rollout Classification Policy Iteration** / classification-based approximate policy
  iteration: Lagoudakis & Parr (2003), Fern–Yoon–Givan, Dimitrakakis & Lagoudakis,
  Farahmand et al. → R1.2b.
- **Multi-pass simulation-based rule selection** in manufacturing: Wu & Wysk (1988) →
  Mouelhi-Chibani & Pierreval (2010) → Shiue, Lee & Su (2020) → R1.2c.
- Adjacent to the **Ulmer / Klapp** line on rollout for dynamic dispatching and routing,
  truncated rollout with a learned value tail, and RL-determined horizons → R5.3.

Corrections required:

- R1.2d — Đurašević & Jakobović is mischaracterised. GP in that literature *generates rules*;
  selectors there are largely supervised, as ours is.
- R1.2e — delete "rollouts are normally used online" and "DAHS inverts the usual deployment".
- R5.2 — add a direct answer to "how does this differ from VFA/RL?" The honest answer is a
  *mechanism* statement, not a novelty claim: DAHS regresses a directly measured per-action cost
  vector, with no bootstrapping and no policy gradient. That is precisely RCPI's argument, and
  the contribution becomes the instantiation plus the controlled comparison.

**Bibliography additions:** Lagoudakis & Parr 2003; Fern/Yoon/Givan; Wu & Wysk 1988;
Mouelhi-Chibani & Pierreval 2010; Shiue et al. 2020; Ulmer (several); Klapp et al. 2018;
Powell (sequential decision analytics). Also **missing but already used in the code**:
Vepsalainen & Morton 1987 (ATC), Smith 1956 (WSPT), Li et al. 2010 (LinUCB).

*Note for the response letter:* R5 states "no C&OR-paper is cited," but reference 3
(Mahmoudinazlou et al. 2025) is *Computers & Operations Research*. Acknowledge the underlying
point — engage with the C&OR / ADP dynamic-dispatching literature — and add the Ulmer/Klapp
line rather than contesting the remark.

### T3. Repositioning the contribution — **your decision** (R1.2e)

R1.2e explicitly asks the authors to choose. Three viable framings:

- **(a) Application + system integration.** Safest, matches the evidence. Contribution = the
  first careful RCPI / multi-pass instantiation for deadline-constrained warehouse dispatching,
  with rule calibration, a switching guardrail, and a controlled comparison to offline RL.
- **(b) Controlled empirical study of training signals.** Reframe around measured cost vector vs
  bootstrapped value vs policy gradient at matched data budgets. The sample-efficiency result
  becomes the legitimate headline, and it survives the F3 metric correction as a *relative*
  result.
- **(c) A narrowed methodological claim.** Keep a method contribution but scope it to what is
  actually new: the multi-sample truncated-rollout label with calibrated distribution plus the
  bias/variance characterisation from M1, positioned as an *extension* of RCPI.

Recommendation: **(b) with elements of (a).**

---

## 4. Remaining reviewer items (bounded work)

| Item | Reviewer | Action |
|---|---|---|
| Position vs order picking; routes/batching/travel time declared exogenous, with justification | R1.1a | Intro + Related Work |
| Warehouse lit review is one sentence | R1.1b | New subsection: problem features, dispatching rules, data-centric methods |
| Fit input distributions to Olist rather than set-then-validate | R1.5b | Extend `experiments/a_realdata_validation.py` with a fitting step |
| Triangular parameters undisclosed | R1.5c | State `[2,5,12]` and `[15,45,90]` in the main text, not only Appendix B |
| Composite cost should be the primary metric | R1.6c | Reframe §6 throughout; state that all learned methods optimise composite cost |
| Offline rollout compute cost | R3.1 | Report total simulated interval-steps and wall-clock on named hardware |
| Pool-size scalability | R3.2 | Discussion + sub-sampling / successive-halving option (concrete once M3 lands) |
| High-load-perishable deep-dive | R3.3 | Rule-selection distribution under saturation + dwell ablation in that cell |
| Ablation supplementary metrics | R3.5 | Add training wall-clock and per-decision inference latency to every ablation row |
| Expand limitations | R3.6 | Subsections: simulation circularity, pool size, single warehouse, no online adaptation, model misspecification |
| DAHS acronym undefined | R1.7a | Define on first use |
| P3 paragraph duplicated | R1.7b | Delete |
| Paragraph titles end ".." | R1.7c | Fix |
| "A reviewer will ask…" | R1.7d | Remove — 3 occurrences (§6.5, §6.9, §8) |
| Reference 2 incomplete | R1.7e | Complete Dokeroglu et al. |

---

## 5. Repo defects found in audit (not reviewer-raised, but a referee could)

1. **Figure 6 contradicts Table 1 on the identical configuration.** E8 cell `1.65_default` is
   byte-identical in config to the default scenario and uses the same 50 seeds, yet reads
   DAHS 0.0048 / greedy_mpc 0.0231 / snapshot_xgb 0.0285 / FEFO 0.0956 against Table 1's
   0.0133 / 0.0313 / 0.0373 / 0.1181. FEFO is a deterministic static rule with no learned
   artifact, so the discrepancy isolates to the simulator or the seed stream, not the models.
   The rebuild resolves this by construction — add a regression test pinning static-rule KPIs
   on fixed seeds so it cannot recur.
2. **§6.6's calibration numbers come from a different model** than Table 1's KPIs — they match
   `runs/data_efficiency/ours_n250_rep0` (ECE 0.063→0.028), not the deployed `runs/phase4`
   (0.059→0.025). Those two runs should be identical (same data, same seed 1337, same
   hyperparameters) and are not (CV soft-xent 0.6768496 vs 0.6762263). Pin library versions;
   add a determinism test.
3. Two of six declared ablations (`no_regime`, `random_ambiguity_filter`) were never run.
4. `paper/manuscript.pdf` link in the README is broken — the file is `dahs_ieee_paper.pdf`.
5. `simpy`, `sb3-contrib`, `tqdm`, `hydra-core` are declared dependencies and never imported.
   The README's smoke test imports `simpy`, which the simulator's own docstring says it
   deliberately does not use. `config.yaml` advertises `batch_prob` / `batch_size_triangular`
   waves citing Boysen et al.; nothing reads them.
6. 116 MB of `runs/data_efficiency/` model artifacts backing a 3.4 KB summary JSON.
7. `baselines/offline_fqi.py:46` hardcodes `PCT_PERISHABLE_IDX = 4` where every other call site
   uses `FEATURE_NAMES.index(...)`.
8. IEEE version lists two authors; README lists one.

---

## 6. Sequencing

| Stage | Content | Compute |
|---|---|---|
| **0** | F1–F5, M1 branch API, M2 baseline, M3 rule library, new metrics module, tests | none |
| **1** | Rule screening + ATC calibration on a disjoint calibration seed set | light |
| **2** | Relabel: corrected model, corrected pool, M-sample rollouts | **the one expensive run** |
| **3** | Retrain regime / ranker / calibrator; τ sweep; data-efficiency sweep | heavy |
| **4** | All baselines incl. rolling-horizon MPC and the RL sensitivity grids | heavy |
| **5** | Scenarios, robustness grid, misspecification sweep, real-data fitting | moderate |
| **6** | Figures, tables, manuscript rewrite, response letter | none |

Stages 0 and 6 are mine. Stages 1–5 need a machine with Python 3.10–3.12 — this one has only
the Windows Store stub, so nothing can be executed or verified here.
