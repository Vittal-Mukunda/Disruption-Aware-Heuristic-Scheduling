---
title: "Sample-Efficient Adaptive Heuristic Selection via Offline Rollout Distillation for Dynamic Warehouse Order Dispatching"
author:
  - Vittal Mukunda
date: 2026
bibliography: references.bib
---

<!--
DRAFT v2 (2026-05-18). Reframed from the v1 "label distribution learning" framing
to "offline rollout distillation". The §6.8 hard-label ablation showed the soft-
vs-hard label choice is immaterial, so the contribution is the rollout-distillation
mechanism itself, not the distributional objective; the title, abstract, §1
contributions, §2, §4.3-4.4, §6.8, §7 and §9 were rewritten accordingly, and
former Limitation #5 (hard-label ablation not run) was removed because the
ablation has now been run and is reported in §6.8. Markdown draft; converts to
Elsevier elsarticle in Phase 9. All quantitative claims are drawn from the frozen
result files under results/ and figures/ (the ground truth); where prose
disagreed with a result file, the result file was used.

DRAFT v3 (2026-05-18). Added a published-competitor baseline: offline_fqi, an
offline reinforcement-learning selector (fitted Q-iteration with FEFO action
masking, the maskable-action-value family of Offline-LD). It trains on the same
logged shifts as DAHS; it is described in §5, added to Tables 1 and 2, evaluated
frozen across all four load scenarios and the twelve-cell robustness grid, and
analysed in the new §6.10. The abstract, §1, §2, §6.5, §7, §9 and Limitation #5
were updated, and references.bib gained the Ernst et al. (2005) fitted-Q citation.
New result files: results/offline_fqi.parquet, results/scenario_*/offline_fqi.parquet,
results/E9/, figures/E9/. No frozen Phase 0-7 result was re-run or re-tuned — this
only adds a baseline.
-->

# Abstract

Dynamic order dispatching in deadline- and perishability-constrained warehouses is
routinely handled by simple priority rules, yet no single rule is best across the
operating conditions a shift passes through. A *selection hyper-heuristic* can pick
the right rule for the current context, but the established ways of learning the
selector are costly: online reinforcement learning is sample-hungry and unstable,
and running a lookahead at every decision is too slow to deploy. We propose
**DAHS**, a selection hyper-heuristic trained by *offline rollout distillation*.
For each decision state in a corpus of simulated shifts we run short
truncated-horizon stochastic rollouts of every candidate rule and record the
per-rule cost vector; we then fit a calibrated gradient-boosted ranker to those
vectors. This amortises an expensive online lookahead into a one-shot, fully
offline supervised signal, and deploys as a single fast forward pass. We prove
that the truncated-rollout cost is a consistent estimator of the full-horizon
cost, with a truncation bias that decays in the rollout horizon, and confirm the
predicted bias–variance trade-off empirically. On 50 held-out simulated shifts,
DAHS attains a 1.33% SLA-breach rate against 3.13% for an analytic
one-step-lookahead controller and 3.73% for an otherwise identical
snapshot-trained ranker, and is Pareto non-dominated across four load scenarios.
A faithful offline reinforcement-learning baseline — fitted Q-iteration with
action masking — trained on the *same* logged shifts reaches only 7.18%, and DAHS
trained on one-tenth the data still outperforms it: the directly measured
rollout-cost signal, not value bootstrapping, is what carries the result.
Critically, DAHS trained on as few as **25 simulated shifts** already outperforms
both baselines at any training budget — a sample-efficiency result we position as
the central contribution. The advantage persists, and widens, when the simulator
is driven by an empirical bursty arrival stream calibrated to a public e-commerce
order trace. An ablation isolating the form of the training label — a soft
cost-distribution versus a hard arg-max — finds the choice immaterial: the working
mechanism is the rollout and its horizon, not the distributional objective.

**Keywords:** hyper-heuristics; dynamic dispatching; rollout; offline distillation;
approximate dynamic programming; warehouse operations.

---

# 1. Introduction

Order dispatching on a warehouse floor is a sequential decision problem under
uncertainty: orders arrive stochastically, each carries a due date and possibly a
perishability constraint, and a small pool of pickers must be assigned work so as
to minimise late and spoiled orders. In practice the decision is delegated to a
*dispatching rule* — first-in-first-out, earliest-deadline-first, weighted
shortest-processing-time, and the like — because rules are transparent, fast, and
require no training. The well-known difficulty is that **no single rule dominates**:
the rule that minimises lateness under light load is not the rule that does so when
the queue is saturated or when a burst of perishable orders arrives. A controller
that *selects* the rule appropriate to the current state — a *selection
hyper-heuristic* [@drake2020hyperheuristics; @dokeroglu2024hyperheuristics] — can in
principle capture the envelope of the pool without abandoning the operational
advantages of rules.

The open question is how to *learn* the selector. The dominant modern answer is
deep reinforcement learning (DRL) [@mahmoudinazlou2025drl; @zhang2024lstmppo].
DRL is attractive but sample-hungry and notoriously unstable on problems where the
per-state advantage of one action over another is small relative to the return
variance — exactly the regime of rule selection, where every rule is a reasonable
policy and the differences are at the margin. A second answer is imitation
learning of an expert dispatcher [@hanjung2025imitation]. Both families face a
structural limitation: DRL never sees the counterfactual cost of the rules it did
*not* take, and imitation needs an expert demonstrator to imitate. The training
signal we use has neither problem — it measures the counterfactual cost of every
rule directly, and needs no expert.

We take a different route. The cost of committing to a rule at a given state can
be *measured directly* by simulation: fix the state, run each candidate rule
forward for a short horizon, and record the cost it incurs. This is a rollout
[@bertsekas2020rollout]. Rollouts are normally used *online* — re-run at every
decision — which is too slow for a warehouse controller. Our approach is to run
rollouts *offline, once*, and use them as a **supervised training signal**. For
each state in a corpus of simulated shifts we roll out every rule, obtain a
per-rule cost vector, and fit a supervised ranker to it. The expensive lookahead
is thereby *distilled* into a cheap function approximator: at deployment the ranker
is a single fast forward pass, and the rollouts live entirely in the training set.
We retain the cost *margin* between rules — not just which rule is best, but by how
much — through a soft, tempered-softmax label by default; an ablation (Section 6.8)
shows the soft form is not essential, and the contribution is the distillation
itself.

This paper makes four contributions.

1. **Offline rollout distillation as a training paradigm.** We run truncated-horizon
   stochastic rollouts of a fixed rule pool *once, offline*, over a corpus of
   decision states, record the per-rule cost vector at each state, and fit a fast
   supervised ranker to it. This amortises an expensive online lookahead into a
   one-shot offline training signal and deploys as a single forward pass —
   avoiding both the per-decision cost of online rollout and the sample-hunger and
   instability of online reinforcement learning.
2. **A consistency result.** We prove (Proposition 1) that the truncated-rollout
   cost is a consistent estimator of the full-horizon rollout cost, with a
   truncation bias bounded by a quantity that decays as the rollout horizon
   lengthens. The bound governs the training signal regardless of how the per-rule
   cost vector is converted into a label, and predicts — confirmed empirically
   (Section 6.4) — that the rollout horizon is the dominant design choice.
3. **A sample-efficiency result.** Because the rollout supplies a directly
   measured, low-variance, per-state target, the ranker learns from very little
   data. DAHS trained on **25 simulated shifts** already outperforms a
   snapshot-trained ranker and an analytic lookahead controller at any budget, and
   a faithful offline reinforcement-learning baseline trained on ten times the
   data. We position sample efficiency, not the headline breach margin, as
   the contribution of record.
4. **Real-data-grounded robustness.** We validate the simulator's input
   distributions against a public e-commerce order trace, and — going beyond a
   passive distributional comparison — re-run the full method comparison with the
   simulator driven by the *empirical* bursty inter-arrival distribution. DAHS
   retains rank one, and its margin over the strongest baseline widens.

One design choice deserves explicit mention up front. DAHS converts each rollout
cost vector into a *soft* training label by a tempered softmax — a
label-distribution representation. An ablation (Section 6.8) shows that replacing
this soft label with its hard arg-max leaves every key performance indicator
statistically unchanged. We therefore frame the contribution as rollout
distillation itself: the distributional form of the label is a detail, not a
mechanism, and we report the ablation that establishes this rather than suppress
it.

The remainder of the paper is organised as follows. Section 2 reviews related
work. Section 3 defines the dispatching problem and the simulator. Section 4
presents DAHS and the consistency result. Section 5 describes the experimental
protocol. Section 6 reports results. Sections 7–9 discuss, list limitations, and
conclude.

---

# 2. Related Work

**Selection hyper-heuristics.** Hyper-heuristics search the space of heuristics
rather than the space of solutions; the *selection* variant chooses, online, which
low-level heuristic to apply [@drake2020hyperheuristics]. Surveys
[@dokeroglu2024hyperheuristics] organise the field around the offline-learning /
online-application paradigm, which DAHS follows: the selector is learned once,
offline, and applied as a fast classifier. Dispatching-rule selection for dynamic
scheduling [@durasevic2022dispatching] is the closest application family. DAHS
differs in its *training signal*: prior selectors are typically trained by
genetic programming, online reinforcement, or hard-label imitation, whereas DAHS
trains a supervised ranker on rollout-derived per-rule cost vectors.

**Rollout and approximate dynamic programming.** A rollout policy improves a base
policy by, at each state, simulating each action followed by the base policy and
choosing the action of least simulated cost [@bertsekas2020rollout]. Rollouts are
a cornerstone of approximate dynamic programming [@simchilevi2021adp] and are
typically truncated to a finite horizon for tractability [@he2024truncatedrollout].
DAHS inverts the usual deployment: rather than running rollouts online at every
decision, it runs them once, offline, to manufacture a supervised training set,
and deploys a cheap function approximator — a *distillation* of the lookahead into
a learned policy. The rollout produces, per state, a vector of per-rule costs; we
encode it for training as a soft label via a tempered softmax — a
label-distribution representation [@geng2016ldl] — although Section 6.8 shows a
hard arg-max encoding performs equivalently. Proposition 1 is a truncation-error
argument in the rollout tradition.

**Deep reinforcement learning for dispatching.** DRL has been applied to dynamic
order picking [@mahmoudinazlou2025drl], order batching [@cheng2024drlhyperheuristic],
and warehouse scheduling [@zhang2024lstmppo], and its scalability for production
scheduling is under active study [@stockermann2025drlscalability;
@tassel2023rljssp]. We include a Proximal Policy Optimization
[@schulman2017ppo] baseline under a matched simulation budget and, separately, at
a 60× budget; Section 6.9 analyses its behaviour. Two recent learning-based
selectors are the closest competitors. Imitation learning of dispatching
decisions [@hanjung2025imitation] trains on the actions of a single expert
dispatcher; DAHS instead distils a *pool* of rules and measures the counterfactual
cost of each, needing no expert. Offline reinforcement learning with maskable
action-value learning [@pluijm2025offlineld] learns a value function from logged
data; DAHS instead regresses a directly measured rollout cost vector, which
sidesteps value bootstrapping and yields a stable supervised objective. We make
this comparison concrete rather than rhetorical: Section 5 includes a faithful
offline fitted-Q baseline with action masking, trained on the same logged shifts,
and Section 6.10 finds it lands well behind DAHS on the primary metric despite
directly optimising the composite cost. A recent
review [@sauer2025mlscheduling] frames bootstrapped self-labelling as an emerging
paradigm for machine learning in scheduling; DAHS is a theoretically grounded
instance of it.

**Warehouse operations and simulation.** Warehouse order-picking design and
control has a long literature [@dekoster2007orderpicking; @roodbergen2006layout;
@boysen2019warehousing; @boysen2025warehousing]. We evaluate in a discrete-interval
simulator; simulation modelling and its verification and validation follow
standard methodology [@law2000simulation; @sargent2013vandv].

**Positioning.** No prior work, to our knowledge, combines (i) truncated
stochastic rollouts of a fixed pool of dispatching rules as the source of an
*offline* supervised training set, (ii) a calibrated supervised ranker as the
deployed selector — sidestepping both the per-decision cost of online rollout and
the instability of online reinforcement learning — and (iii) application to
deadline- and perishability-constrained dynamic warehouse dispatching. DAHS
occupies that intersection.

---

# 3. Problem Setting and Simulator

## 3.1 The dynamic dispatching problem

We consider a single warehouse shift of length $T$ (8 hours). Orders arrive over
the shift; order $o$ has an arrival time $a_o$, a processing time $p_o$, an
absolute due time $d_o$, a perishability flag, and a priority class. A fixed set
of $m$ pickers (10) processes orders; a picker handles one order at a time and is
busy for its processing time. Decisions are taken at the boundaries of $N$ equal
intervals of 15 minutes each ($N = 32$). At each interval boundary the controller
observes the system state and chooses a *dispatching rule* from a fixed pool; the
chosen rule ranks the current order queue, and pickers are assigned greedily down
that ranking until no picker can start an order before the interval ends.

The operator's objective is a composite per-interval cost combining the three
failure modes of deadline-constrained dispatching:

$$ J = \frac{1}{N}\left( W_{\text{breach}}\, n_{\text{breach}} + W_{\text{tardy}} \sum_{o} \tau_o + W_{\text{unfinished}}\, |Q| \right), $$

where $n_{\text{breach}}$ is the number of orders completed after their due time,
$\tau_o = \max(f_o - d_o, 0)$ is the tardiness of order $o$ (completion $f_o$
minus due $d_o$), and $|Q|$ is the count of orders left unprocessed at shift end.
The weights $W_{\text{breach}} = 3.0$, $W_{\text{tardy}} = 0.2$,
$W_{\text{unfinished}} = 0.005$ encode an operator who treats an outright SLA
breach as the dominant cost. These weights are fixed before any learning and are
*not* tuned to the method.

## 3.2 The simulator

The environment is a deterministic discrete-interval simulator. Although the
problem is a discrete-event system, all stochastic quantities — arrival times,
processing times, due-date offsets, perishability and priority draws — are
pre-sampled at construction from a single seeded pseudo-random generator. This
makes a shift *byte-identically reproducible* from its seed, which the rollout
labeller (Section 4.3) requires: it must be able to replay a shift from the start
to reconstruct any decision state exactly.

Arrivals follow a homogeneous Poisson process with base rate 1.65 orders/minute;
processing times and due-date offsets are triangular; a fixed fraction (0.20) of
orders are perishable. The order queue has a capacity of 200. Within an interval
$[t, t{+}15)$ the simulator (i) admits all orders that have arrived, (ii) ranks the
queue with the chosen rule, (iii) assigns each ranked order to the earliest-free
picker with start time $\max(\text{picker free}, a_o, t)$, stopping when no picker
can start before $t{+}15$, and (iv) records end-of-interval key performance
indicators (KPIs). Section 6.7 validates the simulator's input distributions
against a public real-world order trace.

## 3.3 The heuristic pool

The pool contains four dispatching rules: **FIFO** (first-in-first-out by arrival),
**FEFO** (first-expire-first-out, deadline-aware), **WSPT** (weighted shortest
processing time), and **ATC** (apparent tardiness cost, a slack-and-processing
composite). A fifth rule (critical ratio) was screened out during pilot
calibration because it was structurally dominated. The pool is deliberately small
and transparent; DAHS does not invent rules, it *selects* among these four. As
Section 6.1 shows, all four win a non-trivial share of decisions, so the pool is
genuinely diverse and selection is a real problem.

---

# 4. The DAHS Method

## 4.1 Overview

DAHS is an offline-learned, online-applied selection hyper-heuristic. Training
proceeds in four stages: (1) generate a corpus of simulated shifts under a
state-covering behaviour policy; (2) for every decision state, run a
truncated-horizon rollout of each rule and record the per-rule cost vector, from
which a training label is formed; (3) discover a small set of operating *regimes*
and append regime-membership features; (4) fit a calibrated gradient-boosted
ranker to the rollout-derived labels. At deployment a lightweight *switching
controller* wraps the ranker to enforce a perishability constraint and to bound
rule-switching frequency. We describe each stage in turn.

## 4.2 State representation

At each decision boundary the controller observes a 25-dimensional state vector
summarising queue, resource, deadline-pressure, history, and temporal context.
The features include the queue length and its first three lags, mean and maximum
queue age, the fraction of critical and perishable orders, labour utilisation and
the number of busy pickers, mean slack and its dispersion, the count of orders at
risk within 30 minutes, a recent arrival-rate estimate, breach-rate lags, the
interval index and intervals remaining, and the number of arrivals in the current
interval. The complete list is given in Appendix A. To this 25-vector DAHS appends
six regime-membership features (Section 4.5), giving the ranker 31 inputs.

## 4.3 Rollout-informed training labels

The supervisory signal is generated as follows. We simulate 250 training shifts
under a state-covering behaviour policy (a round-robin over the pool, so the
training corpus is not biased toward any one rule's state distribution). This
yields $250 \times 32 = 8000$ decision states. For each state $s_t$ we form a label
over the four rules by **truncated rollout**:

1. Replay the shift from its start to interval $t$, reconstructing $s_t$ exactly.
2. For each rule $h$ in the pool, *commit* to $h$ at $t$, apply the base policy for
   the next $\tau$ intervals, and record the cumulative composite cost
   $\hat{J}^{\tau}_h(s_t)$ incurred over those intervals.
3. Convert the cost vector into a probability distribution by a **tempered
   softmax**:
   $$ p^{\tau}_h(s_t) = \frac{\exp(-\hat{J}^{\tau}_h(s_t)/\beta)}{\sum_{h'} \exp(-\hat{J}^{\tau}_{h'}(s_t)/\beta)}. $$

The temperature $\beta$ is selected once, by a one-dimensional search, so that the
median label entropy falls in a target band $[0.3, 0.7]$ — sharp enough to be
informative, soft enough to retain the cost margin. On the training corpus this
yields $\beta \approx 4.38$ (median entropy 0.63). Two corrections are applied
consistently in both labelling and deployment: when the perishable fraction is
below 0.05 the FEFO mass is zeroed and the distribution renormalised (FEFO is
meaningless without perishables); and, *for the test corpus only*, states whose
maximum label probability is below 0.55 are filtered out as genuinely ambiguous
decisions, leaving 865 of 1600 test states. The horizon is fixed at $\tau = 4$ for
the deployed model; Section 6.4 studies the choice.

By default DAHS uses the soft label above: a state where FEFO and ATC are
near-tied produces a near-uniform target over those two rules, and the ranker is
trained to reproduce that uncertainty rather than to guess. The arg-max of the
cost vector — a *hard* label — is the natural alternative, and Section 6.8 reports
an ablation that finds the two equivalent. We describe the deployed (soft) model
here and treat the label's form as a design choice rather than as a contribution.

## 4.4 A consistency result for truncated rollouts

The deployed model truncates the rollout at $\tau = 4$ of up to 32 intervals. Does
the truncated rollout approximate the cost one would obtain from a full-horizon
rollout? It does, with a controllable bias.

**Proposition 1 (truncated-rollout consistency).**
*Let $\bar{C}$ be an upper bound on the composite cost incurred in any single
interval. Such a bound exists and is finite, because the queue capacity and the
fixed picker count bound the per-interval breach count, total tardiness, and
unfinished-order count. Fix a decision state $s_t$ with $H_t$ intervals remaining
in the shift. For rule $h$, let $J_h(s_t)$ be the full-horizon rollout cost
(commit to $h$ at $t$, base policy for the remaining $H_t$ intervals) and
$\hat{J}^{\tau}_h(s_t)$ the $\tau$-truncated rollout cost, $\tau \le H_t$. Then:*

*(i) the truncation error is non-negative and bounded,*
$$ 0 \;\le\; J_h(s_t) - \hat{J}^{\tau}_h(s_t) \;\le\; (H_t - \tau)\,\bar{C} \;=:\; \Delta_\tau, \qquad \forall h; $$

*(ii) consequently the truncated tempered-softmax label converges to the
full-horizon label as $\tau \to H_t$, with*
$$ \mathrm{KL}\!\left(p^{\infty}(s_t)\,\|\,p^{\tau}(s_t)\right) \;\le\; \frac{2\,\Delta_\tau}{\beta}. $$

*Proof sketch.* (i) The composite cost is a sum of non-negative per-interval
contributions, so truncating the rollout removes a non-negative tail; that tail
spans $H_t - \tau$ intervals, each contributing at most $\bar{C}$. (ii) Write the
label as a softmax of energies $-J_h/\beta$. The truncated energies differ from
the full-horizon energies by at most $\Delta_\tau/\beta$ in absolute value
(part i). The log-sum-exp normaliser is 1-Lipschitz in the supremum norm of its
arguments, so each log-probability shifts by at most $2\Delta_\tau/\beta$; summing
the KL contribution over the distribution gives the stated bound. $\square$

Although part (ii) is phrased for the tempered-softmax label, part (i) bounds the
rollout *cost vector* itself; the consistency therefore transfers to any label
derived from that vector — including the hard arg-max label whose ablation we
report in Section 6.8. In this sense the proposition is a statement about the
rollout, not about the choice of label.

Proposition 1 has two consequences the paper tests. First, the bias $\Delta_\tau$
*decreases monotonically in $\tau$* — longer rollouts give labels closer to the
consistent full-horizon target, so a ranker fit to longer-horizon labels should
fit a more internally consistent signal. Section 6.4 confirms this: cross-validated
soft cross-entropy falls monotonically as $\tau$ goes from 1 to 4. Second, the
bound governs only the *bias*. Estimating $\hat{J}^{\tau}_h$ from a finite number
of stochastic rollouts also incurs *variance*, and that variance grows with
$\tau$, because a longer rollout accumulates more stochastic intervals. DAHS
therefore faces a bias–variance trade-off in $\tau$: Proposition 1 bounds one term,
the rollout estimator's variance drives the other, and the operational optimum is
interior. Section 6.4 finds that optimum at $\tau = 3$.

## 4.5 Regime discovery

Warehouse shifts pass through qualitatively distinct operating regimes — a quiet
opening, a saturated mid-shift, a perishable burst. DAHS makes this explicit. A
Gaussian mixture model is fit to the training-state features; the number of
components is chosen by the Bayesian information criterion over
$K \in \{3,4,5,6\}$, which selects $K = 6$. The fit is checked for stability by
refitting ten times under different seeds and measuring the mean pairwise adjusted
Rand index, which is 0.998 — the regime structure is highly reproducible. The six
soft regime-membership posteriors are appended to the state vector. Regime
discovery is a deliberately lightweight component of the method.

## 4.6 The calibrated ranker

The ranker is a gradient-boosted decision-tree classifier
[@chen2016xgboost] with a four-class soft-probability output. The soft target is
fitted by an inverse-entropy-weighted replication of each training state across the
four classes, which makes the training objective the Kullback–Leibler divergence
between the predicted distribution and the soft label and down-weights states
whose labels are near-uniform (and therefore carry little discriminative signal).
The hard-label variant of Section 6.8 instead uses a standard one-hot
cross-entropy; the rest of the pipeline — feature set, cross-validation,
calibration — is identical. Hyperparameters are selected by 5-fold cross-validation
with folds grouped by shift, so no shift contributes states to both a training and
a validation fold; the search over 18 configurations selects tree depth 4, 500
trees, and learning rate 0.03 (a regularisation-heavy corner), at a cross-validated
soft cross-entropy of 0.677 against a uniform-label baseline of $\log 4 = 1.386$.

Tree ensembles are not probability-calibrated out of the box. DAHS post-processes
the ranker output with isotonic regression fit on a held-out 20% of training
shifts. Calibration quality is reported in Section 6.6; the expected calibration
error improves from 0.063 to 0.028, clearing a pre-registered 0.05 acceptance
threshold.

## 4.7 The switching controller

At deployment the calibrated ranker emits, each interval, a distribution over the
four rules. A thin *switching controller* maps that distribution to an action. It
(i) applies the same FEFO mask used at labelling time; (ii) enforces a minimum
dwell of $T_{\min} = 2$ intervals after a switch, to prevent operationally
disruptive rule thrashing; and (iii) overrides the dwell and switches immediately
when the ranker is highly confident (entropy below half the maximum). We
deliberately frame the controller as a *stability and constraint-enforcement
guardrail*, not a performance driver — and Section 6.8 reports, honestly, that
removing it slightly *improves* the headline KPIs. Its role is to make the policy
deployable (bounded switching, perishability respected), not to win the
comparison; the ablation quantifies the small KPI price of that guardrail.

---

# 5. Experimental Setup

**Test shifts.** All methods are evaluated on the same 50 held-out shift seeds,
disjoint from the 250 training shifts and fixed once. Every reported KPI is a mean
over these 50 shifts.

**Scenarios.** Beyond the default operating point, three scenarios stress the
method: *low load* (reduced arrival rate), *balanced* (moderate load), and
*high-load-perishable* (elevated arrival rate, tighter deadlines, more perishables).
Scenario parameters were fixed before evaluation and are not tuned per method.

**Baselines.** We compare DAHS against: the four static rules (FIFO, FEFO, WSPT,
ATC); **snapshot_xgb**, an ablation identical to DAHS but with the rollout horizon
collapsed to $\tau = 1$ (a one-step / "snapshot" label) — this isolates the value
of the rollout horizon; **greedy_mpc**, an analytic one-step-lookahead controller
that, each interval, simulates each rule for one interval and picks the cheapest —
an *independent*, non-learned controller that does not share DAHS's training
pipeline; **LinUCB**, a contextual bandit; and **PPO** [@schulman2017ppo], a deep
reinforcement learning policy trained under a budget matched to DAHS's (8k
environment steps, *ppo_fair*) and, separately, at a 60× budget (500k steps,
*ppo_full*). Finally, **offline_fqi** is a faithful offline reinforcement-learning
competitor — fitted Q-iteration [@ernst2005fqi] with FEFO action masking, an
instance of the maskable-action-value family of Offline-LD [@pluijm2025offlineld].
It trains on the *same* 250 logged shifts as DAHS, under the *same* round-robin
behaviour policy and per-interval reward, and uses the *same* gradient-boosted-tree
model class as the DAHS ranker; the comparison therefore isolates the training
signal — a directly measured per-rule cost vector versus a single bootstrapped
value — from the function approximator. Its discount and tree hyperparameters
were selected by a 12-configuration search on held-out validation shifts, and
Section 6.10 analyses it.

**Metrics and statistics.** The primary KPI is the SLA-breach rate; we also report
mean tardiness, the composite cost of Section 3.1, throughput, picker utilisation,
and spoilage. Uncertainty is quantified by 10,000-resample bootstrap 95% confidence
intervals; pairwise comparisons use the Wilcoxon signed-rank test with
Benjamini–Hochberg control of the false discovery rate.

---

# 6. Results

## 6.1 The heuristic pool is genuinely diverse

If one rule were best everywhere, selection would be pointless. Figure 1 shows the
per-shift win-rate of each rule — the fraction of decision states at which it is
the cost-minimising choice. All four rules win a material share: FEFO 43%, WSPT
32%, FIFO 15%, ATC 10%. No rule wins a majority, and the winner shifts
systematically with queue state and deadline pressure. Selection is therefore a
real problem, and the ceiling of a perfect selector is well above any single rule.

![Figure 1. Per-shift win-rate of each dispatching rule. All four rules win a
non-trivial share of decisions; no rule dominates.](../figures/E1/diversity_heatmap_shift.png)

## 6.2 Main comparison

Table 1 reports every method on the default scenario. DAHS attains a **1.33%**
SLA-breach rate. The nearest competitors are the analytic lookahead controller
greedy_mpc at 3.13% and the snapshot ($\tau=1$) ranker at 3.73%. DAHS thus improves
on the snapshot ablation by **2.40 percentage points** and on the analytic
controller by 1.80 — the multi-step rollout, absent from the snapshot ($\tau=1$)
ablation, is doing real work. DAHS also attains the lowest composite cost (3.09)
and a low mean tardiness. The faithful offline reinforcement-learning baseline,
offline_fqi, attains 7.18% — behind even the snapshot ablation; Section 6.10
examines why a method that directly optimises the composite cost still loses the
breach metric.

**Table 1.** Default scenario, 50 test shifts. Lower is better for all columns
except throughput and utilisation. DAHS = ours. The ppo_full row coincides
exactly with the FEFO row because the 500k-step PPO policy collapses to always
selecting FEFO (Section 6.9).

| Method | SLA breach | Mean tardiness | Composite cost | Throughput | Picker util. |
|---|---:|---:|---:|---:|---:|
| **DAHS (ours)** | **0.0133** | 0.525 | **3.09** | 721.6 | 0.936 |
| greedy_mpc | 0.0313 | 1.822 | 9.19 | 671.5 | 0.846 |
| snapshot_xgb ($\tau{=}1$) | 0.0373 | 1.589 | 8.77 | 673.7 | 0.851 |
| ppo_fair (8k) | 0.0385 | 0.257 | 3.92 | 740.9 | 0.970 |
| FIFO | 0.0660 | 0.618 | 7.57 | 750.6 | 0.983 |
| LinUCB | 0.0694 | 5.208 | 23.61 | 624.1 | 0.771 |
| offline_fqi | 0.0718 | 0.531 | 7.46 | 734.8 | 0.960 |
| WSPT | 0.0949 | 10.671 | 43.56 | 574.5 | 0.686 |
| FEFO | 0.1181 | 0.997 | 12.60 | 723.8 | 0.947 |
| ppo_full (500k) | 0.1181 | 0.997 | 12.60 | 723.8 | 0.947 |
| ATC | 0.1572 | 1.238 | 16.37 | 721.7 | 0.940 |

DAHS is *Pareto non-dominated*: no baseline beats it on every metric
simultaneously. It does not Pareto-*dominate* the field either — FIFO achieves
higher throughput and utilisation by clearing orders aggressively, at the cost of
more than four times DAHS's breach rate. The honest reading is that DAHS occupies
the corner of the trade-off space the composite objective targets (few breaches,
low cost), and nothing displaces it there.

The multi-scenario picture (Table 2) is consistent. Under low load every method
breaches essentially zero orders — the regime is uninformative. Under balanced and
default load DAHS is rank one. Under the high-load-perishable scenario DAHS's
SLA-breach rate (0.1943) is edged by greedy_mpc (0.1884) by **0.59 percentage
points** — the one cell where DAHS does not lead the breach metric. We report this
plainly. It is a saturation effect: near 19% breach most rules converge, and the
analytic one-step controller's exhaustive per-interval search buys a marginal
breach advantage. DAHS nonetheless retains the lower *composite cost* in that
scenario (100.1 vs 104.8) and the lower mean tardiness (20.7 vs 22.3); the
controller trades a marginal number of breaches for materially lower total cost,
which is exactly what the composite objective asks of it. No method Pareto-dominates
DAHS in any scenario. The offline reinforcement-learning baseline is the one method
that does not converge with the rest under saturation: it rises to a 61.9% breach
rate at high-load-perishable load — more than three times DAHS — the steep
out-of-distribution degradation Section 6.10 examines.

**Table 2.** SLA-breach rate by scenario (50 test shifts).

| Scenario | DAHS | greedy_mpc | snapshot_xgb | offline_fqi | Best static rule |
|---|---:|---:|---:|---:|---:|
| low load | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| balanced | **0.00039** | 0.00546 | 0.00576 | 0.00342 | FIFO 0.00261 |
| default | **0.0133** | 0.0313 | 0.0373 | 0.0718 | FIFO 0.0660 |
| high-load-perishable | 0.1943 | **0.1884** | 0.1949 | 0.6192 | WSPT 0.1965 |

## 6.3 Sample efficiency (the central result)

The 2.40-point breach margin of Section 6.2 is, on its own, a modest empirical
win. The result we ask the reader to weight is *how little data DAHS needs to
reach it*. We retrain DAHS from scratch on training budgets of 25, 50, 100, 150,
and 250 shifts (five independent replications each for the budgets below 250; the
250-shift budget draws the full corpus and is a single deterministic run) and
evaluate on the same 50 test shifts. Figure 4 — the central figure of the paper — plots the outcome.

![Figure 4. Sample efficiency. DAHS SLA-breach rate (left) and composite cost
(right) versus the number of simulated training shifts (mean ± standard deviation
over 5 replications; at the 250-shift budget all five replications draw the
identical full training corpus, so the plotted standard deviation there is zero
by construction). Dashed and dotted lines are the snapshot ranker and the
analytic greedy-MPC controller. DAHS trained on 25 shifts already sits well below
both.](../figures/data_efficiency/data_efficiency_curve.png)

DAHS trained on **25 shifts** attains a 1.44% breach rate — already far below the
snapshot ranker's 3.73% (at 250 shifts) and greedy_mpc's 3.13%. The curve is
essentially flat from 25 to 250 shifts: the full-budget model reaches 1.23–1.33%,
so 90% of the training corpus is, in effect, redundant. This is the operational
signature of rollout distillation: every training state carries a *directly
measured* per-rule target — the rollout cost vector — rather than the noisy,
shift-level return an RL agent must learn from, so the supervised signal is dense
and low-variance and the ranker saturates its learnable structure within a few
dozen shifts. For an operator, this means a deployable controller can be produced
from roughly two and a half hours of simulator time. We regard this sample
efficiency, rather than the breach margin, as the contribution of record.

## 6.4 Rollout horizon: theory meets the curve

Section 4.4 predicted a bias–variance trade-off in the rollout horizon $\tau$.
Table 3 and Figure 5 report DAHS retrained at $\tau \in \{1,2,3,4\}$. Two patterns
appear, exactly as Proposition 1 anticipates. First, the cross-validated soft
cross-entropy — a pure measure of label *fit* — falls monotonically with $\tau$
(0.817, 0.764, 0.709, 0.677). This is the bias term: longer rollouts produce
labels closer to the consistent full-horizon target, so the ranker fits a more
coherent signal. Second, the *operational* KPI does not follow suit monotonically:
SLA breach falls steeply from $\tau{=}1$ to $\tau{=}3$ and then rises slightly at
$\tau{=}4$. This is the variance term: a longer rollout accumulates more stochastic
intervals, so at a fixed rollout count the cost estimate is noisier, and the ranker
can overfit that noise. The operational optimum is interior, at $\tau = 3$.

**Table 3.** Rollout-horizon sensitivity (50 test shifts). Soft cross-entropy is
the cross-validated label fit; lower is better throughout.

| $\tau$ | CV soft cross-entropy | SLA breach | Composite cost |
|---:|---:|---:|---:|
| 1 (snapshot) | 0.817 | 0.0373 | 8.77 |
| 2 | 0.764 | 0.0136 | 2.72 |
| 3 | 0.709 | **0.0105** | **2.31** |
| 4 (deployed) | 0.677 | 0.0133 | 3.09 |

![Figure 5. SLA-breach rate versus rollout horizon. The U-shape — steep
improvement to tau=3, slight regression at tau=4 — is the bias–variance trade-off
of Section 4.4.](../figures/E4/tau_sla_breach_rate.png)

The deployed model uses $\tau = 4$, a horizon fixed before the sensitivity sweep
was run; the sweep shows $\tau = 3$ would have been marginally better still. We
report the deployed model's numbers throughout and do not retro-fit the choice.
The takeaway is structural: even the worst non-trivial horizon, $\tau = 1$, is the
snapshot ablation, and it is 3.5× worse operationally than $\tau = 3$ — the rollout
horizon, the mechanism Proposition 1 analyses, is the single most important design
choice in DAHS.

## 6.5 Robustness across untuned configurations

A reviewer will reasonably ask whether DAHS's advantage is an artefact of the one
operating point at which the simulator was pilot-calibrated. We therefore evaluate
DAHS, greedy_mpc, snapshot_xgb, and the best static rule across a 12-cell grid of
configurations — four arrival rates crossed with three deadline-tightness levels —
of which only one cell was ever used for calibration. No re-tuning is performed on
any cell. Figure 6 shows the SLA-breach grid.

![Figure 6. Robustness grid: SLA-breach rate across 12 untuned configurations
(4 arrival rates x 3 deadline-tightness levels). The method ranking is stable; the
calibrated cell is outlined.](../figures/E8/robustness_grid_heatmap_sla_breach_rate.png)

The relative ranking is stable. DAHS has an SLA-breach rate no worse than the
snapshot ranker in all 12 cells, and no worse than the analytic greedy_mpc
controller in 10 of 12; in the two exceptions (very light load, and the tightest
high-load cell) the methods are within overlapping confidence intervals. DAHS also
has an SLA-breach rate no worse than the offline reinforcement-learning baseline in
all 12 cells, by a margin that widens steeply with load — from a near-tie under
light load to roughly four-fold at the tightest high-load cell (Section 6.10).
Degradation
under heavier load is graceful — DAHS rises from near-zero to roughly 12% breach as
the system saturates — and never collapses, whereas the best static rule (FEFO)
degrades catastrophically (to 70% breach in the hardest cell). The advantage of
DAHS is thus a property of the method, not of the calibrated operating point.

## 6.6 Calibration and interpretability

Isotonic post-processing materially improves the ranker's probability calibration:
the expected calibration error falls from 0.063 to 0.028 and the Brier score from
0.130 to 0.107 (Figure 9). One number moves the other way — the soft
cross-entropy rises from 0.298 to 0.387 — because isotonic regression flattens
over-confident probabilities; this is a known sharpness-versus-calibration
trade-off and we report it rather than suppress it. Calibrated probabilities matter
operationally because the switching controller's entropy gate (Section 4.7) acts on
them.

![Figure 9. Reliability diagrams before and after isotonic calibration. Calibration
pulls the curve toward the diagonal.](../figures/E5/reliability_pre_post.png)

A Shapley-value analysis (Figure 10) shows the ranker's decisions are driven by
operationally sensible features: queue length and its lags, mean slack, the count
of orders at near-term deadline risk, and the interval index dominate the
attribution. The selector is not exploiting an artefact; it keys on the same
quantities a human dispatcher would.

![Figure 10. Global SHAP feature importance for the ranker.](../figures/E5/shap_summary.png)

## 6.7 Real-data grounding

**Distributional validation.** We compare the simulator's input distributions
against the Olist Brazilian e-commerce public order trace [@olist2018dataset]
(~100k orders). Because the trace is measured in days and the simulator in
minutes, we compare distribution *shape* on mean-normalised samples (Figure 7).
The due-date-window distribution matches well (Kolmogorov–Smirnov $D = 0.039$,
normalised Wasserstein distance 0.036). The inter-arrival distribution is in the
right family but the real trace is far more dispersed and heavy-tailed than the
simulator's homogeneous Poisson assumption (coefficient of variation 2.68 vs 1.00;
skewness 11.0 vs 2.0; $D = 0.153$). The order-processing-time comparison *fails*
($D = 0.685$) — but this comparison is not valid: the Olist trace is e-commerce
*order* metadata and carries no warehouse pick-time field, so its
purchase-to-confirmation latency is not a pick time. We state plainly that this
public dataset can validate the arrival and due-date *structure* of the simulator
but cannot validate pick time, for which no public warehouse-floor analogue
exists.

![Figure 7. Simulator input distributions versus the Olist order trace
(mean-normalised; QQ plots and densities).](../figures/A/olist_validation.png)

**Active robustness test.** The distributional comparison is passive — it reports
*how* the simulator differs from reality. We convert it into an active test. The
chief discrepancy is arrival burstiness, so we replace the simulator's Poisson
arrivals with a bootstrap of the *empirical* Olist inter-arrival distribution —
mean-normalised and rescaled so the mean arrival rate is unchanged, injecting the
real coefficient of variation (2.68) and skew (11.0) while holding load fixed — and
re-run the full method comparison. No model is retrained: the frozen DAHS ranker is
evaluated as-is.

Under the bursty real-arrival stream every method degrades — heavy-tailed bursts
transiently overload the queue — but **DAHS retains rank one on every metric**
(Figure 8). More importantly, the *paired* advantage of DAHS over the snapshot
ranker, which cancels common per-shift noise, does not shrink: it is 2.50
percentage points of SLA breach (95% CI [1.68, 3.45]) and 10.0 units of composite
cost (95% CI [6.70, 13.75]) — both wider than under Poisson arrivals (2.40 points
and 5.67 units). DAHS's advantage is therefore not an artefact of the simulator's
idealised arrival process; it survives, and grows under, a realistically bursty
stream calibrated to real data.

![Figure 8. Method KPIs under Poisson versus empirical-Olist bursty arrivals
(95% bootstrap CIs). DAHS holds rank one; its paired margin
widens.](../figures/A2/olist_arrivals_compare.png)

## 6.8 Ablations

We report three ablations, each retraining or reconfiguring a single component of
DAHS and re-evaluating on the 50 test shifts; paired differences against the full
model use the Wilcoxon signed-rank test with Benjamini–Hochberg correction.

**Removing the soft label.** DAHS converts each rollout cost vector into a soft
training target by a tempered softmax (Section 4.3). To isolate whether the
*distributional form* of that target matters, we retrain an otherwise identical
pipeline — same rollout costs, same horizon $\tau = 4$, same 18-configuration
hyperparameter search, same isotonic calibration — on a *hard* label: the one-hot
arg-max of the same cost vector. The choice is immaterial. The hard-label variant
attains a 1.27% SLA-breach rate against the soft model's 1.33% (paired difference
$-0.06$ points; $p = 0.11$, not significant) and a marginally *lower* composite
cost (3.04 versus 3.09; $p = 0.047$). On the primary metric the two are
statistically indistinguishable; on cost the hard label is, if anything, slightly
ahead. We report this plainly: the distributional form of the label is not a
source of DAHS's performance. What does the work is the rollout and, as Section
6.4 shows, its horizon — not whether the per-rule cost vector reaches the ranker as
a soft distribution or as its arg-max. The deployed model retains the soft label
because that choice was fixed before this ablation was run, and we do not retro-fit
it; but the contribution we claim is offline rollout distillation, and this
ablation delimits it honestly.

**Removing isotonic calibration.** Dropping the isotonic post-processing degrades
the SLA-breach rate from 0.0133 to 0.0294 and the composite cost from 3.09 to 7.85
— both highly significant ($p < 0.001$). Calibration is load-bearing, because the
switching controller's entropy gate acts on the predicted probabilities; an
un-calibrated ranker mis-times its switches.

**Removing the switching controller.** Disabling the dwell and the entropy gate
(the ranker's arg-max is followed directly, with the FEFO mask still applied)
changes the default-scenario breach rate from 0.0133 to 0.0118 and the composite
cost from 3.09 to 2.74. Under the Wilcoxon test with BH correction these paired
differences are small but statistically significant ($p_{\text{adj}} = 0.021$ and
$0.001$ respectively). The switching controller therefore does not improve KPIs —
removing it improves them slightly. We retain it deliberately and report this cost
plainly: as Section 4.7 states, its role is to bound rule-switching frequency and
to enforce the perishability constraint — to make the policy operationally
deployable — not to win the comparison. The ablation quantifies the small KPI price
of that guardrail.

## 6.9 On the PPO baseline

A reviewer will ask whether the deep-RL baseline is simply under-trained. The
evidence says the issue is structural, not budgetary. At the matched 8k-step budget,
PPO (*ppo_fair*) does not converge to a useful policy: its deterministic evaluation
policy oscillates between two rules (FIFO and FEFO in roughly equal measure) and
attains a 3.85% breach rate — better than the static rules but well behind DAHS's
1.33%. Increasing the budget 60× to 500k steps (*ppo_full*) makes matters *worse*,
not better: the policy collapses entirely onto a single rule (always-FEFO), with
KPIs identical to the static FEFO baseline (11.81% breach). More training did not
help; it removed what little state-conditioning the smaller-budget policy had.

This is the expected failure mode of policy-gradient RL on this problem. The
per-state advantage of one rule over another is small relative to the variance of
the shift return, so the gradient signal that distinguishes rules is weak; the
policy drifts toward whichever rule has the highest *unconditional* expected return
and, with more updates, commits to it. DAHS sidesteps this precisely because its
training signal is *not* a noisy return — it is a directly measured cost vector,
dense and per-state. The PPO comparison is thus not a horse-race DAHS wins on
tuning; it illustrates why a supervised rollout-distilled signal is the more
suitable instrument for this class of problem.

## 6.10 On the offline reinforcement-learning baseline

The closest published competitor to DAHS is offline reinforcement learning from
logged data [@pluijm2025offlineld]. Section 5 makes that comparison concrete with
*offline_fqi*: fitted Q-iteration [@ernst2005fqi] with FEFO action masking,
trained on the *same* 250 logged shifts as DAHS, under the same round-robin
behaviour policy, the same per-interval reward, and the same gradient-boosted-tree
model class as the DAHS ranker. The only thing that differs is the training
signal — a single bootstrapped value at each visited state–action pair, rather
than the directly measured cost of *every* rule. Because the behaviour policy is
a uniform round-robin, every action is covered at every state, so the
distribution-shift pathologies that motivate conservative offline RL (CQL, IQL)
do not arise and a standard fitted-Q method is the appropriate representative of
the family. Its discount ($\gamma = 0.99$) and tree hyperparameters were selected
by a 12-configuration search on held-out validation shifts.

On the 50 test shifts offline_fqi attains a 7.18% SLA-breach rate, against DAHS's
1.33%. The paired difference — which cancels common per-shift noise — is 5.85
percentage points of breach (95% bootstrap CI [3.85, 8.11]) and 4.36 units of
composite cost (95% CI [2.71, 6.35]); both intervals exclude zero. offline_fqi is
not a weak baseline: it ties DAHS on mean tardiness (0.531 versus 0.525, paired CI
spans zero) and attains a lower composite cost than both the snapshot ranker and
the analytic greedy_mpc controller (7.46 versus 8.77 and 9.19). It is a competent
controller that loses on the metric that matters. The reason is structural, and
it is the one Section 6.9 gives for PPO: an SLA breach is a rare, expensive
event, and a scalar bootstrapped value smears that signal across the bulk cost of
tardiness and queue volume. offline_fqi minimises the bulk cost well — hence its
low tardiness and competitive composite cost — but does not sharply avoid the rare
breach. DAHS's per-rule rollout-cost vector measures the breach-laden cost of each
rule directly, at every state, and the ranker fits it without bootstrapping.

The sample-efficiency gap is starker still (Figure 11). Retrained at training
budgets of 25 to 250 shifts, offline_fqi improves with the training budget but
remains far above DAHS and is still descending at the largest budget (11.55%
breach at 25 shifts, 7.18% at 250). DAHS trained on **25 shifts (1.44% breach)**
outperforms offline_fqi trained on the full **250 shifts (7.18%)** by 5.7
percentage points — a deployable selector from one-tenth the data. offline_fqi is
also far less stable: its breach rate has a cross-replication standard deviation
of 4.5 points at the 25-shift budget, against 0.3 for DAHS — the instability of
value bootstrapping on little data. The offline-RL comparison is therefore not a
horse-race DAHS wins on tuning; like the PPO comparison it shows that a directly
measured rollout-cost signal is the more suitable instrument for this class of
problem, and it answers head-on the comparison a reader of the offline-RL
scheduling literature [@pluijm2025offlineld] will ask for.

![Figure 11. Sample efficiency: DAHS versus the offline reinforcement-learning
baseline. SLA-breach rate (mean ± standard deviation over five replications)
versus the number of simulated training shifts. DAHS is flat near 1.3% from 25
shifts onward; offline_fqi descends from 11.6% but is still well above DAHS at the
full 250-shift budget.](../figures/E9/data_efficiency_offline_fqi.png)

The advantage holds well beyond the default operating point. Evaluated frozen — no
retraining — across the three stress scenarios of Table 2 and the twelve untuned
configurations of the robustness grid (Section 6.5), DAHS has an SLA-breach rate at
or below offline_fqi's in **all four scenarios and all twelve grid cells**: a tie
only under light load, where every method breaches near-zero, and a strict,
widening margin everywhere else. Most telling is the high-load-perishable scenario,
where offline_fqi does not merely lose but *collapses* — its breach rate rises to
61.9%, against 19.4% for DAHS and roughly 19% for greedy_mpc, the snapshot ranker,
and the best static rule. The frozen value function, fit to default-load logged
shifts, transfers poorly to that out-of-distribution saturation regime, whereas
DAHS's rollout-distilled ranker — fit to the very same shifts — degrades gracefully,
in step with the analytic and static baselines. Offline value learning is thus, on
this problem, not only the less sample-efficient instrument but the less robust one.

---

# 7. Discussion

The results support a narrow but well-grounded claim. On deadline-constrained
warehouse dispatching, a selector trained by offline rollout distillation is
*sample-efficient* (Section 6.3), *theoretically consistent* in its training signal
(Section 4.4, Section 6.4), *robust* across untuned operating points (Section 6.5),
and *real-data-grounded* in the sense that its advantage survives a realistically
bursty arrival stream (Section 6.7). The headline SLA-breach margin over the
strongest learned baseline is real but modest (2.40 points); we have been explicit
that the contribution is the *mechanism and its data efficiency*, not the size of
that margin. A faithful offline reinforcement-learning baseline, given identical
data and a fair hyperparameter search, trails DAHS by a wider and statistically
significant margin (Section 6.10); we read that comparison not as a larger
headline number but as controlled evidence that the directly measured
rollout-cost signal — not the choice of learner — is what does the work.

DAHS helps most where rule selection is genuinely contested: moderate-to-high load
with tight deadlines, where the queue state swings the best rule from interval to
interval. It helps least at the extremes — under light load every rule is adequate,
and under extreme saturation the rules converge and an exhaustive analytic
controller can edge it on the breach metric (Section 6.2). The composite cost tells
a more stable story than the breach rate alone: DAHS leads on cost in every
scenario, because the rollout-cost training signal weighs tardiness and unfinished
work, not only outright breaches.

Two deployment properties reinforce the operational case. The first is a **cost
asymmetry**. DAHS runs no simulation at deployment — a decision is a deterministic
forward pass of the regime mixture and the gradient-boosted ranker over the
31-feature state — whereas the analytic controller greedy_mpc simulates every rule
for one interval at *every* decision. DAHS instead pays that lookahead cost once,
offline, at labelling time and amortises it over the deployment. Relative to a
deep-reinforcement-learning policy the asymmetry is one of training data: DAHS
reaches a deployable controller from 25 simulated shifts (Section 6.3), where PPO
does not converge to a useful policy even at 500k environment steps (Section 6.9).
The second property is **auditability of the action**: DAHS emits, each interval,
one of four *named* rules — FIFO, FEFO, WSPT, ATC — that a supervisor already
understands and can check against the visible queue state, rather than an opaque
assignment. We scope this claim carefully: the *selector* is itself a tree ensemble,
interpreted only post hoc through the SHAP attribution of Section 6.6 — it is the
selector's *output*, not its internals, that is transparent by construction. That
is nonetheless a handle on the decision that an end-to-end policy emitting raw
picker assignments cannot offer.

The broader methodological point is that an expensive online computation — rollout
— can be paid *once*, offline, and amortised into a cheap learned approximation,
provided the offline computation is turned into a supervised target rather than a
reinforcement signal. Section 6.10 supplies controlled evidence for that proviso:
an offline reinforcement-learning baseline given the *same* logged shifts, the
*same* model class and a fair hyperparameter search, but trained to bootstrap a
value rather than to regress the measured cost, loses the primary metric by 5.85
percentage points. The rollout yields a per-rule cost vector; a supervised
ranker fit to that vector reproduces the lookahead at a fraction of the deployment
cost. Whether the cost vector is presented to the ranker as a soft distribution or
as its hard arg-max is, on this problem, immaterial (Section 6.8) — the essential
ingredients are the multi-step rollout and the choice of its horizon.

---

# 8. Limitations

We state the limitations of this study plainly.

1. **Simulation-only evaluation.** All KPIs are measured in a simulator. We
   mitigate this by validating the simulator's arrival and due-date distributions
   against a public real-world trace (Section 6.7) and by re-running the comparison
   under empirically calibrated bursty arrivals, but we do not evaluate on a live
   warehouse floor. Pick time in particular has no public real-world analogue and
   is not validated.
2. **A single calibrated operating point.** The simulator was pilot-calibrated at
   one configuration. The robustness grid (Section 6.5) shows the method ranking is
   stable across 11 further untuned configurations, but all of them share the same
   simulator family.
3. **A saturation-load loss.** Under the high-load-perishable scenario DAHS concedes
   0.59 percentage points of SLA breach to the analytic greedy_mpc controller
   (Section 6.2). It retains the lower composite cost, but the breach metric is not
   uniformly won.
4. **Shared-simulator circularity.** The rollout labels and the evaluation use the
   same simulator. This is intrinsic to the method. We mitigate the concern by
   including greedy_mpc — an *independent* analytic controller that does not share
   DAHS's training pipeline — as a baseline; DAHS beats it on the primary metric and
   on cost. A fully independent evaluation environment remains future work.
5. **Reinforcement-learning baselines.** The online PPO baseline collapses on this
   problem (Section 6.9). We mitigate the concern that this is unfavourable to RL
   by construction with a second, *offline* RL baseline (Section 6.10): at the
   default operating point fitted Q-iteration does *not* collapse — it learns a
   competent, non-degenerate policy there that ties DAHS on tardiness — yet still
   loses the primary metric by a wide, statistically significant margin, and it
   degrades far more steeply than DAHS as load rises, collapsing under the
   high-load-perishable scenario. The two RL baselines fail in different ways,
   which makes a tuning artefact an unlikely explanation; but a reviewer may still
   prefer to see a modern conservative offline-RL method (CQL, IQL) compared, which
   we leave to future work.

---

# 9. Conclusion and Future Work

We presented DAHS, a selection hyper-heuristic for dynamic warehouse order
dispatching trained by offline rollout distillation. The method runs
truncated-horizon stochastic rollouts of a fixed rule pool once, offline, records
the per-rule cost vector at each decision state, and fits a calibrated ranker to
those vectors — converting an expensive online lookahead into a one-shot offline
supervised signal that deploys as a single fast forward pass. We proved the
truncated rollout is a consistent estimator of the full-horizon rollout with a
horizon-decaying bias, and confirmed the predicted bias–variance trade-off.
Empirically DAHS is Pareto non-dominated across four load scenarios, robust across
untuned configurations, and ahead of a faithful offline reinforcement-learning
baseline trained on identical data; above all it is — the result we emphasise —
*sample-efficient*: a deployable controller is learned from as few as 25 simulated
shifts, one-tenth the data the offline-RL baseline is given, and its advantage
survives a bursty arrival stream calibrated to real e-commerce order data. An ablation isolating the soft versus hard form of the training label finds
the choice immaterial, locating the contribution squarely in the rollout
distillation itself.

Future work follows the limitations. The most informative next steps are
evaluation on a logged real-warehouse trace or a second, independently built
simulator; extending the approach to a larger or dynamically growing heuristic
pool, where the rollout signal should scale gracefully because it is defined per
rule; and an online variant that adapts the rollout budget $\tau$ to the remaining
horizon — tightening the Proposition 1 bound where it is loosest.

---

# Appendix A. State features

The 25 base state features, by group. *Queue:* queue length; queue length lags 1–3;
mean and maximum queue age; fraction of critical orders; fraction of perishable
orders; number of arrivals in the current interval. *Resources:* labour
utilisation; number of busy pickers; mean recent pickup time. *Deadline pressure:*
number of orders breached so far; number at risk within 30 minutes; mean slack;
slack dispersion; mean remaining processing time; fraction of high-priority orders.
*Arrivals:* recent 60-minute arrival-rate estimate; expected time to next carrier.
*History:* breach-rate lags 1–3. *Temporal:* interval index in shift; intervals
remaining. Six Gaussian-mixture regime-membership posteriors (Section 4.5) are
appended, giving the ranker 31 inputs.

# Appendix B. Configuration and hyperparameters

Simulator: 8-hour shift, 32 intervals of 15 minutes, 10 pickers, queue capacity
200, Poisson arrivals at 1.65 orders/minute, triangular processing and due-date
distributions, perishable fraction 0.20. Cost weights $W_{\text{breach}}=3.0$,
$W_{\text{tardy}}=0.2$, $W_{\text{unfinished}}=0.005$. Labelling: rollout horizon
$\tau=4$, tempered-softmax temperature $\beta\approx4.38$, FEFO mask threshold 0.05,
test-set ambiguity filter at maximum-probability 0.55 (the hard-label ablation of
Section 6.8 replaces the tempered softmax with the one-hot arg-max of the same cost
vector and is otherwise identical). Ranker: gradient-boosted trees, depth 4, 500
trees, learning rate 0.03, selected from an 18-configuration grid by 5-fold
shift-grouped cross-validation; isotonic calibration on a 20% held-out shift split.
Switching controller: minimum dwell $T_{\min}=2$ intervals, entropy-gate threshold
at half the maximum entropy.

# Appendix C. Real-data validation detail

Olist Brazilian e-commerce public dataset [@olist2018dataset], ~100k orders.
Inter-arrival times computed as within-day differences of order timestamps to
remove the multi-day growth trend. Mean-normalised two-sample comparison:
inter-arrival $D=0.153$ (CV 2.68 real vs 1.00 simulated, skew 11.0 vs 2.0);
due-date window $D=0.039$, normalised Wasserstein 0.036; processing-time proxy
$D=0.685$ (not a valid comparison — no warehouse pick-time field in the trace).
Perishable fraction 0.20 simulated vs 0.0099 for the trace's food/drink categories
(a configuration choice, not a generative claim).

# References
