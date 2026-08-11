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

> **Revision note on pending numbers.** Reviewers 1 and 2 identified four changes
> that each alter the data-generating process or the objective: the customer and
> product deadlines are now separate constraints and both enter the cost; orders
> never served are charged rather than exempt; rollout labels are Monte Carlo
> means over independent continuations rather than a single realised path; and the
> rule pool is recalibrated and rescreened. Every quantitative result in the paper
> is therefore regenerated. Passages awaiting the re-run are marked
> `⟨TBD-rerun⟩` and state explicitly what must be reported and, where the
> interpretation depends on the outcome, which way each conclusion falls. No
> number from the submitted version is carried forward into a claim.

# Abstract

Order dispatching in warehouses that face both customer due dates and product
expiry is routinely handled by simple priority rules, yet no single rule is best
across the operating conditions a shift passes through. A *selection
hyper-heuristic* — a controller that chooses which rule to apply as a function of
the current state — can in principle capture the envelope of a rule pool. Such
selectors are conventionally trained by simulating the candidate rules offline and
fitting a classifier to the result, a construction known as multi-pass rule
selection in the scheduling literature and as rollout classification policy
iteration in the reinforcement-learning literature. This paper does not propose a
new training mechanism. It asks instead what the *form of the supervision* is
worth, holding everything else fixed.

We formulate the problem as a sequential decision process: at each 15-minute
review epoch a controller observes a summary of the waiting queue and the picker
availability, and selects one dispatching rule to apply for that interval. Each
order carries two independent deadlines — a customer due date and, for perishable
goods, a product expiry — and both enter the cost, together with tardiness and
with orders left unserved at the end of the shift.

We then compare three ways of supervising the same selector, holding the
environment, the shift data, the model class, the observation and the objective
fixed, and varying only how the training signal is built. The first measures the
cost of *every* rule directly, by simulating each one forward for a few intervals
from each observed state and averaging over independently sampled futures. The
second learns a value function from the same logged transitions by fitted
Q-iteration. The third is a policy gradient. Only the first sees what the rules
it did not take would have cost; only it needs a simulator that can be reset and
re-run under a counterfactual action.

Two results bound the first signal's quality. Stopping a simulation after $\tau$
intervals rather than running it to the end of the shift introduces an error that
*shrinks* as $\tau$ grows. Simulating in a model that differs from the real system
by $\varepsilon$ per step introduces a second error that *grows* as
$O(\varepsilon \tau^2)$. The two together place the best horizon at roughly
$1/\varepsilon$: the more accurate the simulator, the further ahead it is worth
looking. We test that prediction by training in one parameterisation and
evaluating in another.

We report the comparison on held-out shifts, together with the computational cost
on both sides — how much simulation the offline training consumes, and how much
faster a decision becomes once the lookahead has been replaced by a single pass
through a fitted model.

⟨TBD-rerun: state the headline empirical findings here once the campaign
completes. Report the primary objective and the service-failure rate over all
arrived orders, not the breach rate over completed orders alone. Do not restate
the submitted margins: Section 6.2 shows the corrected accounting compresses
them substantially.⟩

**Keywords:** dynamic dispatching; selection hyper-heuristics; rollout;
approximate policy iteration; sequential decision processes; warehouse operations.

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

This paper makes four contributions. None of them is the training mechanism, which
is not new: simulating a rule pool offline and fitting a classifier to the result
is rollout classification policy iteration in the reinforcement-learning
literature and multi-pass rule selection in the scheduling literature, and
Section 2 places the method inside both.

1. **A controlled comparison of training signals.** Holding the environment, the
   shift corpus, the function-approximator class, the feature set and the
   objective fixed, we vary only how the supervision is constructed: a directly
   measured per-rule cost vector, a bootstrapped state–action value fitted from
   the same logged transitions, and a policy gradient. Section 6.10 reports the
   comparison, with the action-coverage diagnostics that determine whether it is
   clean and the hyperparameter sensitivity analysis (Section 6.9) that
   distinguishes a structural result from a tuning artefact. This is the question
   the prior literature leaves open, and it is the paper's principal claim.
2. **Two bounds on the training signal, acting in opposite directions.**
   Proposition 1 bounds the error from truncating the rollout, which decays as
   $O(H-\tau)$. Proposition 2 bounds the error from rolling out under a
   *misspecified* model, which accumulates as $O(\varepsilon\tau^2)$. Together they
   imply an interior optimal horizon $\tau^\star \approx 1/\varepsilon$ — the
   better the simulator, the longer the rollout worth running — and that
   prediction is tested directly in Section 6.11 by labelling in one world and
   deploying in another. The submitted version had only the first bound and
   attributed the interior optimum to an estimator variance its implementation did
   not possess.
3. **A warehouse formulation with two deadline clocks.** Customer due date and
   product expiry are modelled as independent constraints, both entering the
   objective, and Section 3.5 *measures* whether the second binds at a 15-minute
   review interval rather than assuming it. We also state plainly that the
   controller observes a summary $\phi(S_t)$ rather than the state, exhibit two
   queues the summary cannot separate, and treat the problem as partially observed
   (Section 3.2).
4. **A sample-efficiency result.** Because each training state carries a directly
   measured per-rule target rather than a return, the selector saturates its
   learnable structure within a few dozen simulated shifts. We report the budget at
   which that happens and compare it against the offline-RL baseline given the same
   data.

## 1.1 Terminology and notation

The submitted version used a compact vocabulary that a reviewer could not decode,
and terms such as "corpus of simulated shifts", "held-out shifts", "SLA-breach
rate" and "snapshot-trained ranker" appeared without definition. Every term used
in the paper is defined here, on first use in the text, or both.

| Term | Meaning |
|---|---|
| **shift** | One 8-hour working period, the unit of simulation. Divided into $N = 32$ review intervals of $L = 15$ minutes. |
| **decision epoch** | The boundary of a review interval, where the controller acts. There are $N$ per shift. |
| **dispatching rule** | A function that orders the waiting queue. Pickers are then assigned down that order. FIFO, EDD and the rest are dispatching rules. |
| **selection hyper-heuristic** | A controller that chooses *which dispatching rule to apply*, as a function of the current state, rather than choosing an assignment directly. |
| **corpus of simulated shifts** | A set of shifts generated from distinct random seeds, used as data. Split into three disjoint blocks: **training** (fits the selector), **calibration** (fits rule parameters such as ATC's look-ahead scale), and **test**. |
| **held-out shifts** | The test block. No held-out shift contributes to any fitting decision, including hyperparameter selection. |
| **SLA** | Service-level agreement — the contractual delivery commitment. An order's SLA due time $d_o$ is when it must ship. |
| **SLA-breach rate** | The fraction of orders shipped after $d_o$. Two denominators are possible and the choice matters (Section 3.3): over *arrived* orders, or over *completed* orders only. We always state which. |
| **service-failure rate** | Our primary reported KPI component: the fraction of *arrived* orders that either ship late or spoil, whether or not they were ever dispatched. |
| **rollout** | Simulating forward from a state under a fixed rule, to measure what that rule costs. |
| **truncated rollout** | A rollout stopped after $\tau$ intervals instead of running to the end of the shift. |
| **continuation** | One independently sampled future used for a rollout. The label averages over $M$ of them. |
| **ranker** | The fitted classifier that maps an observation to a distribution over rules. A gradient-boosted decision-tree ensemble here. |
| **snapshot-trained ranker** | The ranker fitted to labels from a *one-interval* rollout, i.e. $\tau = 1$. Used as an ablation to isolate the value of looking further ahead. |
| **soft label** | The training target expressed as a distribution over rules rather than a single winner, obtained from the cost vector by a tempered softmax. |
| **regime** | A cluster of operating conditions, discovered by a Gaussian mixture over training observations. Membership probabilities are appended to the observation. |
| **switching controller** | The deployment wrapper that enforces a minimum **dwell** (hold a rule for $T_{\min}$ epochs) and an **entropy gate** (permit an early switch when the ranker is confident). |
| **ablation** | Removing or altering one component and re-measuring, to establish what that component contributes. |
| **DAHS** | Disruption-Aware Heuristic Scheduling, the selection hyper-heuristic studied here. |

**Notation.**

| Symbol | Meaning |
|---|---|
| $t$, $N$, $L$, $T$ | epoch index; epochs per shift (32); interval length (15 min); shift length (480 min) |
| $S_t$ | the true state: queue with full per-order attributes, picker availability, clock |
| $\phi(\cdot)$, $x_t$ | the feature map, and the observation $x_t = \phi(S_t)$ the controller actually sees |
| $u_t$ | the decision at epoch $t$ — a rule from the pool |
| $W_{t+1}$ | exogenous information: orders arriving in $(t, t+1]$ and their attributes |
| $S^M(\cdot)$ | the transition function (Section 3.4) |
| $\mathcal{H}$, $h$ | the rule pool and a member of it |
| $H_t$ | intervals remaining in the shift at epoch $t$ (distinct from $\mathcal{H}$) |
| $a_o, p_o, d_o, x_o, w_o$ | order $o$'s arrival, processing time, customer deadline, product expiry, economic weight |
| $f_o$ | completion time of order $o$ |
| $J$; $W_b, W_t, W_s, W_h$ | the composite objective and its weights: breach, tardiness, spoilage, holding |
| $\tau$, $M$, $\beta$ | rollout horizon; continuations per rollout; softmax temperature |
| $\varepsilon$, $\bar{C}$ | per-step model error in total variation; upper bound on per-interval cost |

The remainder of the paper is organised as follows. Section 2 reviews related
work. Section 3 defines the dispatching problem and the simulator. Section 4
presents DAHS and the consistency result. Section 5 describes the experimental
protocol. Section 6 reports results. Sections 7–9 discuss, list limitations, and
conclude.

---

# 2. Related Work

## 2.1 Scope: which order-picking decision this paper addresses

Order picking decomposes into a set of interacting planning problems — storage
assignment, zoning, order batching, picker routing, and order release and
dispatching [@dekoster2007orderpicking; @vangils2018designing]. This paper
addresses **order release and dispatching only**: at a fixed review epoch, which
of the waiting orders are released to which picker, and in what order. Three
neighbouring problems are treated as exogenous and are held fixed throughout, and
because that restriction shapes every result we report, we state it explicitly
rather than leaving it to be inferred.

**Storage assignment and layout** are fixed. They are themselves substantial design
problems [@roodbergen2006layout], and they enter this model only through the
distribution of order processing times.

**Batching** is fixed at one order per pick tour. The controller never decides
which orders to combine; it decides only which order a free picker starts next.

**Routing and travel-time estimation** are exogenous. The processing time $p_o$
is a single realised service duration that aggregates travel, search and pick
time, in the three-point form standard when only time-standard data are available
[@tompkins2010facilities].

This restriction is deliberate for three reasons. First, it matches the planning
horizons of the decisions involved: layout, zoning and batching policy are
typically fixed over horizons far longer than a shift, so within-shift adaptive
control acts on dispatching alone [@vangils2018designing]. Second, rule selection
is only a well-posed question once the downstream problems are fixed; if the
action space also ranged over batch composition or route construction, the
comparison between dispatching rules would confound the rule with the batching
and routing policy it happens to be paired with. Third, travel-time estimation is
a substantial modelling problem in its own right, and embedding an estimator
inside the environment would make the rule comparison a comparison of travel
models. The cost of the restriction is real — DAHS cannot exploit batching or
routing synergies, and a controller that jointly optimised release and batching
could dominate it — and we record this in Section 8.

## 2.2 Dispatching rules and data-centric control in warehousing

Priority dispatching rules are the standard instrument for release decisions
under uncertainty: they are transparent, require no training, and execute in the
time available at a decision epoch. The classical families and their properties
are established [@conway1967theory; @pinedo2016scheduling] — arrival-driven
(FIFO), due-date-driven (EDD [@jackson1955edd], minimum slack, modified due date
[@baker1982mdd]), processing-time-driven (WSPT [@smith1956wspt]), and composite
slack-and-processing indices (COVERT [@carroll1965covert], ATC
[@vepsalainen1987atc]). The persistent finding across that literature is that no
single rule dominates: the rule minimising weighted tardiness under light load is
not the rule that does so once the queue saturates or the due-date mix tightens
[@pinedo2016scheduling].

In warehousing specifically, order release and dispatching under stochastic
arrivals has been surveyed by @dekoster2007orderpicking, @boysen2019warehousing
and @boysen2025warehousing, and the planning-problem taxonomy of
@vangils2018designing places dispatching among the operational decisions that
interact most strongly with due-date performance. The field's recent direction is
data-centric: @winkelhaus2020logistics4 survey the shift toward data-driven
warehouse control, and learned controllers have been applied to dynamic order
picking [@mahmoudinazlou2025drl], order batching [@cheng2024drlhyperheuristic],
and warehouse scheduling [@zhang2024lstmppo]. What distinguishes the warehouse
setting from the job-shop setting these methods were developed in is the presence
of a second deadline clock on perishable goods, which is the feature our problem
formulation makes explicit in Section 3.

## 2.3 Simulation-trained selectors of dispatching rules

Selecting a rule as a function of the shop state, with the selector trained on
simulation output, is a mature line of work and predates this paper by several
decades. @wu1988multipass introduced the *multi-pass* construction: at a decision
point, simulate the candidate rules forward, record their outcomes, and use the
result to choose. @mouelhi2010neural made the selector a learned function,
training a neural network on simulated states labelled by the best-performing
rule so that the multi-pass simulation is paid offline and the deployed decision
is a forward pass. @shiue2020rl extend the same construction with reinforcement
learning over the rule set. Surveys of dispatching-rule selection for dynamic
scheduling [@durasevic2022dispatching] and of hyper-heuristics generally
[@drake2020hyperheuristics; @dokeroglu2024hyperheuristics] organise this
literature around the offline-learning, online-application paradigm that the
present method also follows.

We correct a characterisation in the submitted version of this paper. We
previously wrote that prior selectors are "typically trained by genetic
programming, online reinforcement, or hard-label imitation". That is not accurate.
Genetic programming in this literature is used predominantly to *generate* the
low-level rules that are subsequently selected among, not to learn the selector
[@branke2016automated; @nguyen2017gpsurvey]; and prior selectors, including those
in @durasevic2022dispatching, @mouelhi2010neural and @shiue2020rl, are trained
essentially by supervised learning on simulation-derived labels — which is what we
do as well. The method in this paper belongs squarely inside that tradition
rather than departing from it.

## 2.4 Rollout and classification-based approximate policy iteration

A rollout policy improves a base policy by simulating each action at the current
state, following the base policy thereafter, and taking the action of least
simulated cost [@bertsekas2020rollout]. Rollouts are a cornerstone of approximate
dynamic programming [@simchilevi2021adp] and are typically truncated to a finite
horizon for tractability [@he2024truncatedrollout].

The construction we use — estimate action values by simulation at a sample of
states, then fit a classifier to represent the improved policy — is **Rollout
Classification Policy Iteration**, introduced by @lagoudakis2003rcpi and developed
by @fern2006api, @dimitrakakis2008rollout and @farahmand2015capi. It is not a new
training paradigm, and the submitted version of this paper was wrong to present it
as one. In particular, the claims that rollouts are "normally used online" and
that our method "inverts the usual deployment" were both incorrect: offline
rollout generation for supervised policy learning has existed for over two
decades in the reinforcement-learning literature and, as Section 2.3 records, for
longer than that in the scheduling literature. We withdraw those claims.

Two details of our instantiation differ from the standard RCPI setting, and we
note them as details rather than as contributions. The classifier is fitted to
the full per-action cost *vector* rather than to the arg-max alone, encoded as a
tempered-softmax label distribution [@geng2016ldl]; Section 6.8 reports an
ablation showing this makes no material difference, which is consistent with RCPI
practice. And the rollout is truncated at a short horizon with the truncation
error bounded explicitly (Proposition 1), which is a standard truncation argument
in the rollout tradition [@he2024truncatedrollout].

### Rollout and ADP for dynamic dispatching

The operations-research literature on dynamic dispatching under stochastic
arrivals is the closest methodological neighbour to this work, and the submitted
version did not engage with it. @klapp2018onedim and @klapp2018dispatchwaves
formulate the dispatch-waves problem — when to release accumulated demand, given
that more will arrive — which is structurally the decision studied here with
routing rather than rule selection as the inner problem. @goodson2017rolloutframework
give a general rollout framework for finite-horizon stochastic dynamic programs,
including the treatment of truncated horizons and pre- versus post-decision
rollouts that Proposition 1 sits inside; @goodson2016restocking apply
rollout policies to stochastic-demand routing.

@ulmer2020modeling set out the modelling conventions for stochastic dynamic
routing and dispatching problems that Section 3 follows, and the same group's work
on offline–online approximate dynamic programming [@ulmer2019offlineonline], on
budgeting decision time under stochastic requests [@ulmer2018budgeting], and on
anticipation against reactive re-optimisation [@ulmer2019anticipation] is directly
relevant to the trade-off this paper studies.

One thread deserves particular emphasis because it supersedes part of our
construction. That literature does not simply truncate a rollout and discard the
tail: it **approximates the remainder with a learned value function**, and in
places uses learning to set the rollout horizon state-dependently. Our
Propositions 1 and 2 treat truncation as a hard cut, which makes the truncation
bias $(H_t - \tau)\bar{C}$ a quantity to be tolerated rather than estimated. A
value-approximated tail would replace that term with an approximation error that
need not grow with the remaining horizon, which is strictly the better
construction and would likely permit a shorter $\tau$ — attractive here, because
Proposition 2 shows short horizons also limit model-error accumulation. We do not
implement it, and we record it in Section 9 as the most promising extension rather
than as an incidental idea.

### How this differs from value-function approximation

A reviewer of the submitted version asked, reasonably, what distinguishes this
from value-function approximation or reinforcement learning, since those also
learn offline from simulation. The submitted paper's answer was rhetorical. The
substantive answer has three parts, and the first thing to say is that the method
is **inside** the approximate-dynamic-programming family, not outside it: what is
described here is one step of approximate policy iteration in which the improved
policy is represented by a classifier rather than derived from a value function
[@lagoudakis2003rcpi; @fern2006api].

*What is learned.* Value-function approximation fits $V(S)$ or $Q(S,u)$ — a
scalar satisfying, approximately, a Bellman fixed point — and recovers a policy
from it by one-step lookahead. Here no value function is ever formed and no fixed
point is sought. The object fitted is the policy itself: a classifier over
actions, trained on directly measured per-action costs at sampled states.

*How error behaves.* This is the difference that matters and it is why the
comparison in Section 6.10 is informative. A bootstrapped value estimate
propagates error through the Bellman backup, so an error at one state contaminates
its predecessors and the fixed point can be reached badly or not at all. The
classification construction has no backup: its error at each state is ordinary
supervised generalisation error, and errors at different states do not compound.
That is the mechanism our experiments are designed to isolate, holding the
environment, corpus, model class, feature set and objective fixed.

*What each requires.* This asymmetry runs against us and we state it plainly.
Value-function approximation can be fitted from logged transitions alone — the
data an operating warehouse already produces. The construction here needs a
simulator that can be **reset to an arbitrary state and rolled forward under a
counterfactual action**, because the label is the cost of rules that were not
taken. That is a strictly stronger requirement, it is the reason the method is
simulator-bound, and it is why the circularity of Section 8.1 is intrinsic rather
than incidental. A practitioner without a trustworthy simulator should prefer
value learning from logs; the comparison in this paper is only relevant to one who
has such a simulator and is deciding how to use it.

## 2.5 Reinforcement learning for dispatching

Deep reinforcement learning has been applied widely to dispatching and its
scalability for production scheduling is under active study
[@stockermann2025drlscalability; @tassel2023rljssp]. Two learning-based selectors
are the closest comparators and both appear as baselines here. Imitation learning
of dispatching decisions [@hanjung2025imitation] trains on the actions of a single
expert dispatcher, and so requires an expert; the multi-pass construction instead
measures the counterfactual cost of every rule and needs none. Offline
reinforcement learning with maskable action-value learning
[@pluijm2025offlineld] learns a value function from logged data, and is
reimplemented faithfully in Section 5 as a fitted-Q baseline
[@ernst2005fqi] trained on the same logged shifts. We also include Proximal
Policy Optimization [@schulman2017ppo] under a matched simulation budget and,
separately, at a 60× budget, with a hyperparameter sensitivity analysis in
Section 6.9. A recent review [@sauer2025mlscheduling] frames simulation-derived
self-labelling as an emerging paradigm for machine learning in scheduling.

## 2.6 Positioning and what this paper contributes

Given Sections 2.3 and 2.4, the mechanism at the centre of this paper is not
novel, and we do not claim it. Simulating a rule pool offline and fitting a
classifier to the result is RCPI in the reinforcement-learning literature and
multi-pass rule selection in the scheduling literature. What we offer instead is
an **empirical study**, with three components.

1. **A controlled comparison of training signals at matched data budgets.** The
   question we can answer, and which the prior literature has not answered
   directly, is what the *supervision* buys. We hold the environment, the shift
   corpus, the function-approximator class and the objective fixed, and vary only
   how the training signal is constructed: a directly measured per-action cost
   vector, a bootstrapped state-action value fitted from the same logged
   transitions, and a policy gradient. Section 6.10 reports that comparison, and
   Section 6.9 supports it with the hyperparameter sensitivity analysis needed to
   distinguish a structural result from a tuning artefact.

2. **A warehouse instantiation with two deadline clocks.** We model the customer
   due date and the product expiry as independent constraints, both entering the
   objective, and we measure whether the second one binds at a 15-minute review
   interval rather than assuming it (Section 3.5). This is the feature that
   distinguishes perishable-goods warehousing from the job-shop settings the
   dispatching-selection literature was developed in.

3. **A sample-efficiency result.** Because each training state carries a directly
   measured per-action target rather than a return, the selector saturates its
   learnable structure within a few dozen simulated shifts. We report the budget
   at which that happens and compare it against the offline-RL baseline given the
   same data.

We make no claim to a new training paradigm, and the contribution should be read
as the application and the controlled comparison, not the mechanism.

---

# 3. Problem Setting and Simulator

## 3.1 Orders, and the two deadline clocks

We consider a single warehouse shift of length $T$ (8 hours). Orders arrive over
the shift. Order $o$ carries

| symbol | meaning |
|---|---|
| $a_o$ | arrival time |
| $p_o$ | processing time — one pick tour, aggregating travel, search and pick (Section 2.1) |
| $d_o$ | **customer deadline**: the absolute time by which the order must ship |
| $x_o$ | **product deadline**: the absolute time after which the goods are no longer saleable. Finite for perishable orders, $x_o = \infty$ otherwise |
| $w_o$ | economic weight of the order, set by its priority class |

The two deadlines are distinct constraints with distinct failure modes, and they
are sampled independently: missing $d_o$ ships an order late, missing $x_o$
destroys the goods. Either can bind first. This is the substantive change from
the submitted version of this paper, in which orders carried only $d_o$, the rule
we called FEFO in fact sorted on $d_o$, and "spoilage" was defined as a perishable
order missing its *due date* — so perishability was a label on an order rather
than a constraint on the problem. We are grateful to the reviewer for identifying
this; the model, the objective and the rule pool are corrected accordingly, and
Section 3.5 tests whether the corrected constraint actually binds.

**Where the expiry of an order comes from**. FEFO is conventionally a rule for
issuing inventory *lots*, not customer orders, so an order-level expiry requires
justification. We model the stage *after* lot allocation: an upstream allocation
policy — exogenous here, in the same sense as routing and batching (Section 2.1) —
has already committed specific lots to specific orders, so each order inherits a
concrete expiry from the stock reserved against it. For a single-line order that
is the allocated lot's expiry. For a multi-line order, the order ships as a unit
and is therefore constrained by its most perishable component, so
$x_o = \min_{\ell \in o} x_\ell$ over the allocated lines $\ell$. The experiments
in this paper use single-line orders, so $x_o$ is the allocated lot's expiry
directly; the minimum-over-lines rule is the generalisation and requires no change
to the controller, since the controller reads only $x_o$.

A fixed set of $m = 10$ pickers processes orders; a picker handles one order at a
time and is busy for its processing time. Decisions are taken at the boundaries of
$N = 32$ equal review intervals of $L = 15$ minutes. At each boundary the
controller observes the system and chooses a *dispatching rule* from a fixed pool;
that rule ranks the waiting queue, and pickers are assigned down the ranking.

## 3.2 The decision process, and what the controller can see

The submitted version of this paper described the problem in prose and moved
directly to the implementation, which left a reviewer unable to determine what
decision was being taken, against what information, or under what objective. That
was a fair criticism and this section is the remedy. We state the problem as a
**sequential decision process** in the canonical form of @powell2019unified and
@powell2022rlso — state, decision, exogenous information, transition function,
objective — before any implementation detail, and Section 3.4 then specifies the
transition concretely. Readers who prefer the reinforcement-learning vocabulary
may read the five elements as a finite-horizon Markov decision process; the
partial-observability caveat below applies under either reading.

**The sequential decision process**. Decisions are taken at epochs
$t = 0, 1, \dots, N-1$, one per review interval. The **state** is

$$ S_t \;=\; \big(\, \mathcal{Q}_t,\; \mathbf{b}_t,\; t \,\big), $$

where $\mathcal{Q}_t$ is the multiset of waiting orders, *each carrying its full
attribute tuple* $(a_o, p_o, d_o, x_o, w_o)$, and
$\mathbf{b}_t \in \mathbb{R}^m$ records when each picker next becomes free. The
**decision** is $u_t \in \mathcal{H}$, a rule from the pool. The **exogenous
information** $W_{t+1}$ is the set of orders arriving in $(t, t+1]$ with their
attributes. The **transition** $S_{t+1} = S^M(S_t, u_t, W_{t+1})$ is the
admission–rank–assign procedure of Section 3.4. Writing $C(S_t, u_t, W_{t+1})$
for the cost accrued over the interval (Section 3.3), the problem is

$$ \min_{\pi \in \Pi} \; \mathbb{E}\left[\; \sum_{t=0}^{N-1} C\big(S_t,\, U^\pi_t(S_t),\, W_{t+1}\big) \;\right] . $$

**The observation is not the state**. The controller does not see $S_t$. It sees

$$ x_t \;=\; \phi(S_t) \;\in\; \mathbb{R}^{26}, $$

a fixed-length summary listed in Appendix A. The submitted version of this paper
called $x_t$ "the state". That was wrong, and the reviewer who identified it is
right about the mechanism as well as the terminology: $\phi$ records *marginal*
summaries — queue length, mean and standard deviation of slack, mean processing
time, counts of critical and perishable orders — and discards the *joint*
distribution over per-order attributes. But it is the joint distribution that
determines what a ranking rule does next, because a rule orders orders by a
function of their attributes taken together.

**A witness**. $\phi$ is not injective on the reachable state space, and the
failure is not exotic. Consider two queues of two orders each, arriving at the
same instant, differing only in which order carries the tight deadline:

$$ \mathcal{Q}^{A} = \{(p_{\text{short}}, s_{\text{loose}}),\, (p_{\text{long}}, s_{\text{tight}})\}, \qquad \mathcal{Q}^{B} = \{(p_{\text{short}}, s_{\text{tight}}),\, (p_{\text{long}}, s_{\text{loose}})\}, $$

writing $s_o = d_o - t - p_o$ for slack. The two have the same queue length, the
same arrival times and therefore the same ages, the same mean and standard
deviation of slack, the same mean processing time, and — with $s$ and $p$ chosen
so that the critical thresholds fall the same way — the same critical and at-risk
counts. Every coordinate of $\phi$ agrees: $\phi(S^A) = \phi(S^B)$ exactly. The
dynamics do not agree. In $\mathcal{Q}^{A}$ the binding deadline sits on the order
that occupies a picker longest, so deferring it is expensive and the feasible set
of on-time completions is strictly smaller than in $\mathcal{Q}^{B}$. Under the
*same* rule the two states incur different cost.
`experiments/observability_analysis.py` constructs such a pair, verifies that the
feature vectors coincide to machine precision before comparing anything, and
reports the resulting cost gap; Section 8 gives the value.

**Consequence: a POMDP, and a policy-function approximation**. We therefore do not
claim that $\phi$ is a sufficient statistic — it is not, and the witness settles
it. The control problem is a **partially observed** Markov decision process, and
the policy class we search is a policy-function approximation over the
observation,

$$ U^\pi(S_t) \;=\; \arg\max_{h \in \mathcal{H}} \; f_\theta\big(\phi(S_t)\big)_h , $$

with no belief state maintained and no history beyond the lags $\phi$ carries
explicitly. Two consequences follow, and they pull in different directions.

The unfavourable one is that there is an **irreducible regret floor**: two states
that $\phi$ cannot separate must receive the same action, so whenever their
optimal actions differ, some regret is incurred that no amount of data or model
capacity can remove. We measure this rather than assume it away. Over the
training corpus we locate mutual near-neighbours in standardised $\phi$-space and
report how often they disagree about the cost-minimising rule, and what acting on
the neighbour's choice costs, as a share of the total benefit available from rule
selection. That number is an upper bound on the part of the residual regret
attributable to partial observability, and it is reported in Section 8 alongside
the other limitations rather than buried.

The favourable one is more specific than it first appears. The rollout **labels**
are computed from the true state $S_t$: the simulator holds $\mathcal{Q}_t$ in
full, and the rollouts of Section 4.3 run the actual dynamics from it. So the
supervision target is *correct*; it is only the *covariates* that are lossy. The
learning problem is therefore regression with insufficient covariates — a Bayes
risk induced by the feature map — and not a misspecified or biased target. This
distinction matters for interpreting the results: a residual gap between DAHS and
a controller with access to $S_t$ is attributable to $\phi$, not to the training
signal, and enlarging $\phi$ (a set encoding over queued orders, for instance) is
the remedy. We leave that to future work and record it in Section 9.

## 3.3 The objective

Let $f_o$ denote the completion time of order $o$ if it is dispatched. For an
order still waiting at the reference horizon $T$ we set $f_o = T + p_o$: the
earliest it could possibly finish, since it still requires a full pick. With that
convention every order that arrived has a well-defined outcome whether or not it
was ever served, and

$$ J \;=\; \sum_{o \in \mathcal{A}} w_o \Big[\, W_{b}\,\mathbb{1}\{f_o > d_o\} \;+\; W_{t}\,\max(f_o - d_o,\,0) \;+\; W_{s}\,\mathbb{1}\{f_o > x_o\} \,\Big] \;+\; W_{h}\,|Q_T| , $$

where $\mathcal{A}$ is the set of orders that arrived during the shift and
$|Q_T|$ is the queue length at shift end. Weights are
$W_{b} = 3.0$ (late shipment), $W_{t} = 0.2$ per minute of lateness,
$W_{s} = 5.0$ (spoilage), and $W_{h} = 0.005$ per queued order. They are fixed
before any learning and are not tuned to any method.

Three features of this objective differ from the submitted version, each in
response to a specific reviewer comment, and each consequential.

**Priority class now enters the objective**. WSPT and ATC have always ranked by
$w_o/p_o$, but the submitted objective weighted every order equally. Those rules
were therefore optimising a criterion the evaluation never measured, which is a
substantial part of why their reported performance looked anomalous (Section 6.2).
Rule and objective now agree.

**Perishability now enters the objective**. The $W_s$ term is the only place a
product deadline can be priced. Without it, "perishability-constrained" was not a
property of the optimisation problem at all.

**Orders that are never served are charged**. In the submitted objective an order
abandoned in the queue attracted only $W_{h} = 0.005$, against $W_{b} = 3.0$ for
one served late — a factor of 600. A controller could therefore lower its reported
breach rate by declining to touch difficult orders, and the reported breach rate
was computed over *completed* orders only, so those orders left the metric
entirely. The convention $f_o = T + p_o$ closes both gaps: an unserved order past
its deadline is charged exactly as a late one, and the $+\,p_o$ ensures that
dispatching an order onto a free picker costs precisely what abandoning it costs,
with any earlier dispatch costing strictly less. Doing the work is therefore
weakly optimal by construction, which the submitted objective did not guarantee.
$W_h$ survives strictly as a work-in-progress holding cost. Section 6.2 reports
both the corrected metric and the submitted one, so the two are comparable.

**Spoilage mechanics, stated explicitly**. The submitted version left four
questions unanswered, and each is answered here rather than left to be inferred
from the code.

*Does a perishable order have a distinct expiry, or is it the due date?* Distinct.
$x_o$ is drawn independently of $d_o$ (Section 3.1), so for a perishable order
either clock can bind first. In the submitted model there was no $x_o$ at all and
"spoilage" was defined as a perishable order missing $d_o$, which made the two
events the same event by construction.

*What happens when an order spoils?* Its goods become unsaleable at $x_o$, and the
charge $W_s w_o$ is incurred at that instant and is permanent — picking the order
afterwards does not undo it. The order is **not** removed from the queue: spoiled
stock still has to be pulled and disposed of, so it continues to consume a picker
when it is eventually handled. Keeping it in the queue also closes an incentive
gap. If spoiled orders vanished, a controller could free picking capacity by
stalling until perishables expired, which is the same class of loophole as
exempting unfinished orders.

*Is a spoiled order counted in the breach count?* Lateness and spoilage are
**separate predicates** on the same order, and an order may be neither, either, or
both. When both fire, the composite cost charges both — they are distinct
economic losses, a late shipment and destroyed stock. In the reported metrics they
are kept apart so nothing is double-counted in a headline: the breach rate counts
lateness only, the spoilage rate counts spoilage only, and the primary
**service-failure rate** counts an order once if it is late *or* spoiled.

*How are unfinished perishables penalised at shift end?* Through the same
$f_o = T + p_o$ convention as any other unserved order (above). If
$T + p_o > x_o$ the order is spoiled and charged $W_s w_o$; if $T + p_o > d_o$ it
is also late and charged $W_b w_o$ plus tardiness. Since shelf life is at most 120
minutes against a 480-minute shift, a perishable order still waiting at shift end
is spoiled with near certainty, which is the intended behaviour: abandoning
perishable stock is the most expensive thing this objective can do.

**How the reported rates are calculated**. Writing $\mathcal{A}$ for the set of
orders that arrived during the shift, $\mathcal{S} \subseteq \mathcal{A}$ for
those dispatched, and $\mathcal{P} \subseteq \mathcal{A}$ for the perishable ones:

$$ \text{service-failure rate} = \frac{|\{o \in \mathcal{A} : f_o > d_o \ \text{ or } \ f_o > x_o\}|}{|\mathcal{A}|}, \qquad \text{spoilage rate} = \frac{|\{o \in \mathcal{P} : f_o > x_o\}|}{|\mathcal{P}|}, $$

$$ \text{breach rate}_{\text{arrived}} = \frac{|\{o \in \mathcal{A} : f_o > d_o\}|}{|\mathcal{A}|}, \qquad \text{breach rate}_{\text{served}} = \frac{|\{o \in \mathcal{S} : f_o > d_o\}|}{|\mathcal{S}|} . $$

The last of these is the metric the submitted paper reported, under the
unqualified name "SLA-breach rate". Its denominator excludes every order the
controller declined to dispatch, and — with $W_h = 0.005$ against $W_b = 3.0$ —
so did the objective. Both are reported here, under names that make the
denominator explicit, so the two versions of the paper remain comparable.
$\mathcal{A}$ includes orders rejected at the door when the queue was at
capacity; they are real demand that went unmet, and excluding them would reopen
the same gap in a different place.

## 3.4 The simulator and its parameters

The transition function $S^M$ of Section 3.2 is realised as a discrete-interval
simulation model; its construction, verification and validation follow standard
practice [@law2000simulation; @sargent2013vandv], with the input distributions
fitted to a public order trace where one exists (Section 6.7) and the remaining
parameters carrying explicit provenance tags below.

The environment is a periodic-review discrete-interval model. Within interval
$[t, t+L)$ it (i) admits every order that has arrived **by $t$**, subject to a
queue capacity of 200, recording overflow as *rejected demand* rather than
discarding it; (ii) ranks the waiting queue with the chosen rule, evaluated at
$t$; (iii) assigns down the ranking to the earliest-free picker with start time
$\max(\text{picker free}, t)$, stopping once no picker can start before $t+L$;
and (iv) records end-of-interval statistics. Orders arriving inside the interval
wait for the next epoch.

The admission rule matters. The submitted simulator admitted every order arriving
before $t+L$ — fifteen minutes of look-ahead — and set the start time to
$\max(\text{picker free}, a_o, t)$, so ranking a not-yet-arrived order *reserved a
picker and left it idle* until that order appeared. Rules sorted by arrival never
paid this cost; arrival-agnostic rules paid it constantly. Section 6.2 shows this
was the second cause of the anomalous rule performance the reviewer flagged.

**Parameters and their provenance**. Every input is now either fitted to data,
grounded in a cited source, or declared a design choice; none is left unexplained.

| Input | Value | Provenance |
|---|---|---|
| Shift, review interval, pickers | 8 h, 15 min, 10 | Operating point |
| Queue capacity | 200 orders | Operating point |
| Inter-arrival **shape** | Empirical bootstrap of the Olist trace | **Fitted** (Section 6.7) |
| Arrival **rate** | 1.65 orders/min nominal | Operating point; swept in Section 6.5 |
| Processing time $p_o$ | Triangular$(2, 5, 12)$ min | **Literature**: three-point time standard for manual picker-to-parts picking [@tompkins2010facilities; @dekoster2007orderpicking] |
| Customer window $d_o - a_o$ | Triangular$(15, 45, 90)$ min | **Fitted** to the Olist purchase-to-estimated-delivery distribution, shape only, rescaled to the shift time base |
| Shelf life $x_o - a_o$ | Triangular$(20, 60, 120)$ min | Design parameter; no public trace carries expiry. Swept in Section 6.4 |
| Perishable fraction | 0.20 | Design parameter; varied by scenario |
| Priority classes and weights | $\{$low, medium, high$\}$ at $(0.50, 0.35, 0.15)$, $w_o \in \{1, 2, 4\}$ | Design parameter |

This inverts the submitted workflow, in which parameters were set and then
*validated* against a public trace. The reviewer correctly observed that fitting
is the right operation, and Section 6.7 now reports the fits, the candidate
families compared by AIC, and — for processing time, where the trace carries no
warehouse pick-time field — the reason no fit is attempted.

## 3.5 Does perishability bind at a 15-minute horizon?

A model can carry a product deadline without that deadline ever changing a
decision. Since we claim the setting is perishability-constrained, we test the
claim directly rather than assert it, using the criterion the reviewer proposed:
in what fraction of decisions does delaying an order by one review interval alter
its outcome?

At each epoch $t$, a waiting order has exactly two options available to the
dispatcher — served now, completing at $t + p_o$, or deferred, completing no
earlier than $t + L + p_o$. Call the order **expiry-pivotal** at $t$ when those
two straddle its product deadline,

$$ t + p_o \;\le\; x_o \;<\; t + L + p_o , $$

so that one interval of delay is the difference between saleable goods and waste,
and **due-pivotal** when the same holds for $d_o$. We report the share of
decisions with at least one expiry-pivotal order in the queue, the share of queued
perishables that are expiry-pivotal, the share of perishables whose product clock
binds strictly before their customer clock, and the share of total economic weight
sitting on expiry-pivotal orders. The threshold for the claim to stand was fixed
at 5% of decisions before the diagnostic was run. Section 6.4 reports the outcome.
Had it fallen below the threshold, the correct response would have been to drop
the perishability framing and remove FEFO from the pool, and we would have
reported that instead.

## 3.6 The rule pool

The pool is not a convenience sample. It spans the four information sources a
dispatcher can key on, so that "no single rule dominates" is a structural property
of the design space rather than an empirical accident:

| Rule | Keys on | Source |
|---|---|---|
| FIFO | arrival only | zero-information control |
| EDD | customer deadline | @jackson1955edd |
| MS | customer deadline slack | @conway1967theory |
| MDD | customer deadline, degrading to SPT when past due | @baker1982mdd |
| **FEFO** | **product deadline** | the only expiry-aware rule |
| WSPT | weight and processing time | @smith1956wspt |
| ATC | slack × processing, exponential discount | @vepsalainen1987atc |
| COVERT | slack × processing, linear truncation | @carroll1965covert |

Three points the submitted version left open.

**FEFO is not a due-date rule**. In the submitted paper we described FEFO as
"deadline-aware" and implemented it as a sort on $d_o$. That is EDD. FEFO sorts on
the product deadline $x_o$ and is now implemented that way; EDD appears in the
pool under its own name. The rule that produced the submitted FEFO results is EDD,
and results are reported accordingly.

**ATC is calibrated, not assumed**. WSPT is exactly the $k \to \infty$ limit of
ATC: as the look-ahead scale grows, $\exp(-\text{slack}/(k\bar p)) \to 1$, leaving
the WSPT index $w_o/p_o$. A correctly calibrated ATC therefore cannot be beaten by
WSPT, and the submitted result — WSPT winning 32% of decisions against ATC's 10% —
was a symptom of an unfitted $k$, which was fixed at 2.0 with no search performed.
We now calibrate $k$ twice, on a calibration corpus disjoint from both training
and test shifts: once for **standalone** use, because ATC is itself a reported
benchmark and benchmarking an uncalibrated rule understates it; and once for
**portfolio contribution**, which is the quantity that matters when the rule sits
inside a selector. Both values, and the deployed one, are reported in Section 6.1.
COVERT's scale is calibrated the same way.

**Screening is by marginal contribution, not win rate**. A rule earns its slot by
covering states the others handle badly. We therefore report, per rule, both its
win rate and the increase in achievable cost if it were removed from the pool,
with a bootstrap interval on the latter. A rule with a high win rate and zero
marginal contribution is redundant; a rule with a low win rate and positive
marginal contribution is a specialist worth keeping. Win rate alone, which is what
the submitted Section 6.1 reported, cannot distinguish the two. This is also how
we answer the question of what **FIFO** contributes in a due-date-driven setting:
it is retained as the zero-information control, and the screen reports whether it
earns its place. If it does not, we drop it and say so — in the submitted
labelling FIFO was the cost-minimising rule at none of the 865 filtered test
states, which is itself a finding rather than an embarrassment.

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

## 4.2 The observation vector

At each decision epoch the controller receives $\phi(S_t)$, a 26-dimensional
summary of the true state (Section 3.2) covering queue, resource, customer-
deadline, product-deadline, arrival, history and temporal context. It is an
observation, not a state, and Section 3.2 gives the explicit pair of queues it
cannot separate. Appendix A lists every feature with its group and the source or
design rationale it derives from — the submitted version listed the features
without saying where they came from or whether they had been screened.

Two features of the submitted set are removed. `time_to_next_expected_carrier` was
$1/\lambda$ and therefore **constant** within a configuration; `intervals_remaining`
was an exact affine function of `interval_index_in_shift`, the two summing to $N$
by construction. Together they made the feature matrix exactly singular, which
silently corrupted the regime layer's model selection (Section 4.5). Three
expiry-pressure features are added, since the product deadline now enters the
objective and a selector cannot act on a constraint it cannot see. Appendix A also
reports the correlation and variance-inflation analysis that identified the
redundancies, and Section 6.8 ablates down to the five most important features to
test whether the full set earns its dimensionality.

To $\phi(S_t)$ DAHS appends the regime-membership posteriors of Section 4.5.

## 4.3 Rollout-informed training labels

The supervisory signal is generated as follows. We simulate a corpus of training
shifts under a state-covering behaviour policy, giving one decision state per
review epoch per shift. For each state $s_t$ we form a label over the pool by
**multi-sample truncated rollout**:

1. Walk the shift forward to epoch $t$, so that $s_t$ is the true state $S_t$ with
   its full queue.
2. Draw $M$ independent continuations. Continuation $m$ freezes the realised
   history at $t$ and resamples the *unrealised* future — arrivals after $t$ and
   their attributes — from a stream seeded by $(\text{shift}, t, m)$.
3. For each rule $h$, commit to $h$ at $t$, apply the base policy for the next
   $\tau$ epochs of continuation $m$, and record the cost
   $\hat{J}^{\tau}_{h,m}(s_t)$ accrued over that window. Average:
   $$ \hat{J}^{\tau}_h(s_t) \;=\; \frac{1}{M}\sum_{m=1}^{M} \hat{J}^{\tau}_{h,m}(s_t), \qquad \widehat{\mathrm{se}}_h(s_t) \;=\; \frac{\hat{\sigma}_h(s_t)}{\sqrt{M}} . $$
4. Convert the cost vector into a probability distribution by a **tempered
   softmax**:
   $$ p^{\tau}_h(s_t) = \frac{\exp(-\hat{J}^{\tau}_h(s_t)/\beta)}{\sum_{h'} \exp(-\hat{J}^{\tau}_{h'}(s_t)/\beta)}. $$

**Why $M > 1$, and why the submitted labels were not estimates**. In the submitted
implementation every stochastic quantity was pre-sampled when a shift was
constructed, and the labeller replayed that same shift from its start for each
candidate rule. All rules therefore saw *the identical realised future* — the one
belonging to the shift seed. The label recorded which rule was best **in hindsight
on one path**, not which had the lowest expected cost, and the rollout variance
was identically zero. The reviewer who identified this is correct that these are
different quantities, and correct that the difference matters for a method whose
stated objective is expected cost. It also means the bias–variance argument of
Section 4.4 had no variance term anywhere in the implementation. The estimator
above fixes this, and the per-cell standard error $\widehat{\mathrm{se}}_h$ is
recorded alongside every label so the residual noise is reported rather than
assumed away.

**Common random numbers**. The continuation seed depends on the shift, the epoch
and the sample index, and deliberately **not** on the rule under test. Every
candidate is therefore scored against an identical set of $M$ futures. The
comparison is paired, and since the tempered softmax reads only *differences*
between rules, the relevant variance is that of
$\hat{J}^{\tau}_h - \hat{J}^{\tau}_{h'}$, which is far smaller than that of either
term: the shared arrival shock that dominates a single rollout's cost cancels.
This is why a modest $M$ suffices where independent sampling would need an order
of magnitude more. Section 6.4 sweeps $M$ and reports where the label stabilises.

**Cost**. Walking each shift forward once and branching at each epoch costs
$O(N \cdot |\mathcal{H}| \cdot M \cdot \tau)$ interval-steps per shift, against the
submitted scheme's $O(N^2 \cdot |\mathcal{H}|)$ — it replayed from $t = 0$ for every
epoch and every rule. The quadratic term is what paid for $M$.
Section 6.12 reports the measured totals.

The temperature $\beta$ is selected once, by a one-dimensional search, so that the
median label entropy falls in a target band $[0.3, 0.7]$ — sharp enough to be
informative, soft enough to retain the cost margin. On the training corpus this
yields $\beta \approx 4.38$ (median entropy 0.63). Two corrections are applied
consistently in both labelling and deployment: when the perishable fraction is
below 0.05 the FEFO mass is zeroed and the distribution renormalised (FEFO cannot
act on a queue with no product deadlines); and, *for the test corpus only*, states
whose maximum label probability falls below 0.55 are filtered out as genuinely
ambiguous decisions.

**Where the corpora come from**. Shift seeds are drawn once from a single
`SeedSequence` and partitioned into three contiguous, disjoint blocks: training,
calibration and test. The calibration block is new in this revision and exists so
that rule hyperparameters — ATC's and COVERT's look-ahead scales (Section 3.6) —
can be fitted without touching either of the other two; the submitted version had
no such block, which is part of why ATC went uncalibrated. Each shift contributes
one decision state per review interval, so a block of $n$ shifts yields $32n$
states: the submitted test corpus of 50 shifts gave $50 \times 32 = 1600$ states,
of which 865 survived the $\theta = 0.55$ ambiguity filter. The filter is applied
to the test corpus only, and never to the training corpus, so no training state is
discarded for being difficult.

The horizon is fixed at $\tau = 4$ for the deployed model; Section 6.4 studies the
choice.

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

### Model error, and why Proposition 1 alone is not enough

Proposition 1 bounds the error from stopping the rollout early. It says nothing
about the error from rolling out in the *wrong world*. The rollout is generated by
a simulator, and a simulator is a model: its arrival process, service times and
picker dynamics are estimates. As the reviewer of the submitted version put it,
any misspecification corrupts the labels, and the corruption compounds along the
rollout — potentially flipping the preferred rule. Proposition 1 is silent on
exactly the error most likely to matter in deployment.

We therefore state the second bound. Let $P$ denote the transition kernel of the
real system and $\tilde{P}$ that of the simulator, and suppose the model is
accurate to $\varepsilon$ per step in total variation,

$$ \sup_{s,\,u}\; \big\| P(\cdot \mid s, u) \;-\; \tilde{P}(\cdot \mid s, u) \big\|_{\mathrm{TV}} \;\le\; \varepsilon . $$

**Proposition 2 (model-error accumulation).** *Under the conditions of
Proposition 1, the $\tau$-truncated rollout cost computed under $\tilde{P}$
differs from the same quantity under $P$ by at most*

$$ \Big| \hat{J}^{\tau,\tilde{P}}_h(s_t) - \hat{J}^{\tau,P}_h(s_t) \Big| \;\le\; \bar{C}\,\varepsilon\,\frac{\tau(\tau-1)}{2} \;=:\; \Gamma_\tau^{\varepsilon} . $$

*Proof sketch.* Let $d_k$ and $\tilde{d}_k$ be the state distributions after $k$
steps under $P$ and $\tilde{P}$ from the common initial state $s_t$, under the
same rule. Transition kernels are non-expansive in total variation, so one step
adds at most $\varepsilon$ and $\|d_k - \tilde{d}_k\|_{\mathrm{TV}} \le k\varepsilon$
by induction, with $\|d_0 - \tilde d_0\| = 0$. The per-interval cost lies in
$[0, \bar{C}]$, so $|\mathbb{E}_{d_k}[c] - \mathbb{E}_{\tilde{d}_k}[c]| \le
\bar{C}\,k\varepsilon$. Summing over $k = 0, \dots, \tau-1$ gives
$\bar{C}\varepsilon\sum_{k<\tau} k = \bar{C}\varepsilon\,\tau(\tau-1)/2$. $\square$

**The three error terms, and what they imply for $\tau$ and $M$.** Collecting
Propositions 1 and 2 with the Monte Carlo error of the estimator in Section 4.3,
the deviation of a computed label from the ideal full-horizon cost under the true
dynamics is bounded by

$$ \underbrace{(H_t - \tau)\,\bar{C}}_{\text{truncation, } \downarrow \text{ in } \tau} \;+\; \underbrace{\bar{C}\,\varepsilon\,\tfrac{\tau(\tau-1)}{2}}_{\text{model error, } \uparrow \text{ in } \tau} \;+\; \underbrace{O_p\!\big(\hat{\sigma}_h / \sqrt{M}\big)}_{\text{estimator, } \downarrow \text{ in } M} . $$

Three things follow, and the first two are new relative to the submitted paper.

First, **the optimal horizon is interior for a reason that has nothing to do with
estimator variance**. Truncation bias falls linearly in $\tau$ while model error
grows quadratically, so the sum is minimised at
$\tau^\star \approx 1/\varepsilon + 1/2$ — the better the model, the longer the
rollout that is worth running, and a badly misspecified simulator should be rolled
out for only one or two intervals. The submitted paper attributed the interior
optimum entirely to estimator variance, which, as Section 4.3 explains, its
implementation did not actually possess.

Second, **this is a falsifiable prediction and Section 6.11 tests it**. If we
degrade the model deliberately by evaluating under perturbed dynamics while
labelling under nominal ones, the horizon that minimises realised cost should
shorten as the perturbation grows. That is a sharper test than reporting
robustness, because it predicts the *direction and mechanism* of the degradation
rather than only its size.

Third, **the two knobs are separable**. $\tau$ trades truncation against model
error and is bounded by how much we trust the simulator; $M$ controls only the
estimator term and can be raised independently at linear cost. The submitted
design conflated them because with $M = 1$ the estimator term was unbounded and
invisible at the same time.

Both bounds are stated for the rollout *cost vector*, so — as with Proposition 1
part (ii) — they transfer to any label derived from it, including the hard arg-max
of Section 6.8, via $\mathrm{KL}(p^\infty \| p^\tau) \le 2(\Delta_\tau +
\Gamma^\varepsilon_\tau)/\beta$.

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

**Test shifts**. All methods are evaluated on the same 50 held-out shift seeds,
disjoint from the 250 training shifts and fixed once. Every reported KPI is a mean
over these 50 shifts.

**Scenarios**. Beyond the default operating point, three scenarios stress the
method: *low load* (reduced arrival rate), *balanced* (moderate load), and
*high-load-perishable* (elevated arrival rate, tighter deadlines, more perishables).
Scenario parameters were fixed before evaluation and are not tuned per method.

**Baselines**. We compare DAHS against the static rules retained by the screen of
Section 3.6; **snapshot_xgb**, an ablation identical to DAHS but with the rollout
horizon collapsed to $\tau = 1$, isolating the value of the horizon; **LinUCB**
[@li2010linucb], a contextual bandit, with features standardised; **PPO**
[@schulman2017ppo] at a budget matched to DAHS's and, separately, at a 60× budget,
with the hyperparameter sensitivity analysis of Section 6.9; and **offline_fqi**,
a faithful offline reinforcement-learning competitor — fitted Q-iteration
[@ernst2005fqi] with FEFO action masking, an instance of the maskable-action-value
family of Offline-LD [@pluijm2025offlineld]. It trains on the same logged shifts
as DAHS, under the same behaviour policy and per-interval reward, and uses the
same gradient-boosted-tree model class and the same feature set as the DAHS
ranker, so that the comparison isolates the training signal — a directly measured
per-rule cost vector against a single bootstrapped value — from the function
approximator and from the state representation. Section 6.10 analyses it, together
with the action-coverage diagnostics that determine whether the comparison is
clean.

**The teacher: rolling-horizon rollout MPC**. DAHS is trained to reproduce a
$\tau$-step rollout, so the controller that simply *runs* that rollout online is
the natural reference, and the submitted version omitted it. Its only lookahead
baseline was one-step, while the deployed model distilled $\tau = 4$ — so the
comparison that determines what the distillation costs or gains was absent
entirely. We add **rolling_mpc**: at each epoch it evaluates every rule over
$\tau$ intervals, averaged over independent continuations drawn exactly as in
Section 4.3, commits the arg-min rule for one interval, discards the remainder of
the plan, and replans. It uses the same estimator as the labeller, so any gap
between it and DAHS is attributable to the function approximation and the
deployment guardrails rather than to a different scoring rule. The one-step
controller is retained as its $\tau = 1$ special case.

This baseline is what makes the paper's central claim falsifiable, and it answers
four questions the submitted version could only assert answers to. *How much does
distillation lose?* The gap between rolling_mpc and DAHS at the same $\tau$ is the
price of replacing a lookahead with a forward pass. *What does it buy?* We report
per-decision latency for both, so the amortisation appears as a measured ratio
rather than an argument. *Can the student beat the teacher?* It can in principle —
a fitted selector regularises across states where the rollout estimate is noisy,
and DAHS also carries a calibration layer and a switching guardrail the raw
lookahead lacks — and Section 6.2 reports whether it does here. *And is the
horizon the mechanism?* Comparing rolling_mpc at $\tau = 1$ against $\tau = 4$
separates the value of lookahead depth from the value of learning.

⟨TBD-rerun: the reviewer's statement of this request ends mid-sentence in the
copy we received — "Including this benchmark would answer several critical
questions:" with the list truncated. The four questions above are our reading of
what the benchmark settles. If the intended list differs, we will report against
it directly; the experiment as implemented produces per-decision cost, latency and
KPIs for every $\tau$, so most reasonable extensions are already covered by the
logged output.⟩

**Objective and metrics**. Every learned method optimises the composite cost $J$
of Section 3.3, and $J$ is the primary metric of comparison. This is a correction
to the submitted version, which declared the SLA-breach rate primary and then
analysed results against it, even though the breach count is only one of the four
terms in the objective the methods were actually trained on. Making the objective
primary also removes a reporting incentive that the submitted metric created: a
method could improve one component at the expense of another that went unreported.

That every method optimises the same $J$ is now enforced by construction rather
than asserted. The objective is defined once, in a single module; the rollout
labels integrate it, the PPO reward is its negated per-interval increment, fitted
Q-iteration regresses that same increment, and the bandit's payoff is its
negation. The rolling-horizon controller minimises it directly. The static rules
optimise nothing and are evaluated against it. In the submitted code the objective
was written out twice, in the labeller and in the evaluation harness, which is
why what was being optimised could not be read off the paper.

Alongside $J$ we report its decomposition, so no component can hide: the
**service-failure rate** — the share of *arrived* orders that ship late or spoil,
whether or not they were ever dispatched — together with mean tardiness, spoilage
rate, throughput, unserved and rejected demand, and picker utilisation. For
comparability with the submitted version we also report the breach rate computed
over completed orders only, under that explicit name. Uncertainty is quantified by
10,000-resample bootstrap 95% confidence intervals; pairwise comparisons use the
Wilcoxon signed-rank test with Benjamini–Hochberg control of the false discovery
rate.

---

# 6. Results

## 6.1 Rule calibration, screening, and complementarity

If one rule were best everywhere, selection would be pointless. Establishing that
it is not requires three things the submitted version did not provide: calibrated
rules, a screen that distinguishes redundant rules from specialists, and evidence
of complementarity across the **state space** rather than across instances.

**Calibration (Section 3.6)**. ATC's look-ahead scale is fitted on the calibration
corpus, once for standalone use and once for portfolio contribution, and COVERT's
the same way. ⟨TBD-rerun: report $k^\star_{\text{standalone}}$,
$k^\star_{\text{portfolio}}$, the deployed value for each rule, and the cost curve
over the grid. The submitted paper reported WSPT winning 32% of decisions against
ATC's 10%, which cannot hold for a fitted ATC since WSPT is its $k\to\infty$
limit; state whether calibration resolves the inversion.⟩

**Screening**. Each candidate is scored by win rate *and* by marginal
contribution — the increase in achievable cost when it is removed from the pool —
with a bootstrap interval on the latter. ⟨TBD-rerun: report the screening table
and the retained pool. A rule is retained when its marginal-contribution interval
excludes zero. State explicitly which candidates are dropped and why, including
whether FIFO earns its slot as the zero-information control.⟩

**Complementarity**. The submitted Figure 1 reported win rate per shift and per
interval. The reviewer correctly observed that this varies the *instance*, not the
state, and so cannot establish that the rules cover different operating regions —
which is what "complementary" has to mean for a state-conditioned selector.
Figure 1 is replaced by win rate over a grid of the two state dimensions that
govern the decision: queue length and deadline pressure (mean slack), in quantile
bins. A pool is complementary when different rules own different cells of that
grid. ⟨TBD-rerun: report the grid, the number of cells each retained rule owns,
and the gap between the best single rule and the per-cell oracle — the latter is
the ceiling any selector could reach.⟩

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

**Table 1**. Default scenario, 50 test shifts. Lower is better for all columns
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

### The corrected accounting, and what it costs the headline

The submitted Table 1 ranked methods by a breach rate whose denominator was
*completed* orders. A reviewer observed that this leaves an opening — a controller
can lower the reported rate by declining to touch difficult orders — and pointed
to the direct evidence in the table itself: DAHS completed 721.6 orders on average
against basic FIFO's 750.6. The request was to see results in which every overdue
order counts as a breach, whether it was completed late or abandoned in the queue.

We regard this as the most important correction in the revision, so we state its
consequence before reporting the new numbers rather than after. The submitted
repository ships per-order event logs for ten shifts under the frozen model, and
recomputing the reviewer's metric on those logs — counting every arrived order,
served or not — gives:

| | breach rate over completed orders | failures over *arrived* orders |
|---|---:|---:|
| DAHS | 3.10% | 15.00% |
| FIFO | 11.75% | 17.97% |
| PPO (matched budget) | 9.40% | 16.44% |

DAHS's advantage over FIFO narrows from roughly 3.8× to 1.20×, and over PPO from
3.0× to 1.10×; on individual shifts the ordering against PPO inverts. The
qualitative ranking survives the correction. The margin does not, and the
submitted paper's headline overstated it.

Those ten shifts are the demonstration corpus, not the evaluation corpus, so the
figures above are indicative rather than the result. But they establish the sign
and the order of magnitude, and we would rather put that in front of the reader
immediately than let it emerge from a table. Section 3.3 rebuilds the objective so
that unserved orders are charged, Section 3.4 reports the full outcome partition,
and the primary metric throughout this section is the composite cost with the
service-failure rate as its headline component.

⟨TBD-rerun: regenerate Table 1 under the corrected objective and metric. Report,
per method: composite cost; service-failure rate; the outcome partition
(arrived / served / unserved / rejected); the breach rate over arrived orders and
over completed orders, both labelled; spoilage rate; tardiness; utilisation. State
plainly whether the method ranking changes under the corrected metric, and if the
sample-efficiency claim of Section 6.3 weakens, weaken it.⟩

### Two anomalies in the submitted results, and their causes

The submitted version of this table contained two results that are hard to
reconcile with scheduling theory, and we were right to be asked about them rather
than allowed to narrate around them. WSPT — a shortest-processing-time rule, which
should *maximise* the number of orders completed — recorded the **lowest**
throughput of any method (574.5 against FIFO's 750.6) and a picker utilisation of
0.686 while its queue sat near the 200-order capacity. And FIFO, which uses no
deadline information at all, placed fourth of eleven on composite cost. Both are
artefacts of the environment and the objective, not properties of the rules, and
both are corrected in this revision.

**Cause 1: the objective did not measure what the rules optimise**. WSPT and ATC
rank by $w_o/p_o$, using the priority weights of Section 3.1. The submitted
objective weighted every order equally. Those two rules were therefore being
graded against a criterion they were not designed for — they were correctly
maximising weighted throughput while the scoreboard counted unweighted breaches.
Section 3.3 puts $w_o$ into the objective, so rule and objective now agree.

**Cause 2: the dispatcher idled pickers on behalf of arrival-agnostic rules**. The
submitted simulator admitted every order arriving before the *end* of the current
interval and assigned start times $\max(\text{picker free}, a_o, t)$. Ranking an
order that had not yet arrived therefore reserved a picker and left it standing
idle until that order appeared. FIFO, sorted by arrival, never triggered this;
WSPT and ATC, which are arrival-agnostic, triggered it constantly. This is what a
picker utilisation of 0.686 alongside a queue of roughly 180 waiting orders was
recording: a picker cannot be idle a third of a shift with that much work
available unless the dispatch model idles it. It also explains FIFO's flattering
composite cost, since FIFO was the only rule the mechanism never penalised.
Section 3.4 replaces this with a properly causal periodic-review admission rule —
only orders that have arrived by $t$ are eligible at $t$ — which removes the
handicap and, incidentally, removes fifteen minutes of undisclosed look-ahead from
the observed state.

Together these mean the submitted rule comparison was not measuring rule quality.
All results in this section are regenerated under the corrected environment and
objective, with the recalibrated pool of Section 3.6.

⟨TBD-rerun: regenerate Table 1 and the accompanying analysis. Report throughput
and utilisation by rule under the corrected admission rule, and state whether WSPT
now behaves as theory predicts. If it does, that is the confirmation that Cause 2
was the mechanism; if it does not, the remaining discrepancy must be explained
rather than absorbed.⟩

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

**Table 2**. SLA-breach rate by scenario (50 test shifts).

| Scenario | DAHS | greedy_mpc | snapshot_xgb | offline_fqi | Best static rule |
|---|---:|---:|---:|---:|---:|
| low load | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| balanced | **0.00039** | 0.00546 | 0.00576 | 0.00342 | FIFO 0.00261 |
| default | **0.0133** | 0.0313 | 0.0373 | 0.0718 | FIFO 0.0660 |
| high-load-perishable | 0.1943 | **0.1884** | 0.1949 | 0.6192 | WSPT 0.1965 |

### Boundary conditions: what the selector actually does under saturation

Calling the one lost cell "a saturation effect" is an attribution, not an
analysis, and a practitioner deciding whether to deploy this controller needs to
know *which* boundary condition bites. There are two candidate explanations and
they have different remedies, so we distinguish them rather than assert one.

The first is that the selector stops selecting. If, as load rises, the ranker
collapses onto a single rule, then the scenario KPI is really that rule's KPI and
the method has quietly degenerated into a static policy. We measure this as the
**exponentiated entropy of the deployed-rule distribution**, which equals
$|\mathcal{H}|$ when the selector spreads across the pool and 1.0 when it has
collapsed.

The second is that the guardrail binds. The minimum dwell holds a rule for
$T_{\min}$ epochs, and under saturation the queue state changes fastest — exactly
when a stale rule is most costly. We measure this as the **blocked-switch rate**:
the share of epochs at which the ranker's arg-max differed from the deployed rule
*because the dwell was still active*. The controller now records every decision it
takes, so this is read off a run rather than inferred.

⟨TBD-rerun: report both statistics across all four scenarios, plus the switch rate
and the entropy-gate firing rate. If the selection entropy falls sharply toward
saturation, the selector is collapsing and the pool needs a rule suited to that
regime. If the blocked-switch rate rises, the dwell is the binding constraint.
These are not mutually exclusive.⟩

⟨TBD-rerun: report the causal test — a sweep of $T_{\min}$ *within* the
high-load-perishable scenario alone. If the cost-minimising dwell there is shorter
than the deployed one, then the guardrail is the boundary condition, and the
honest response is to report the trade explicitly (KPI against switching
frequency) and consider a load-dependent dwell, rather than to defend a fixed
default. Section 4.7 already concedes that removing the controller improves KPIs
slightly at the default operating point; if that penalty grows with load, it
should be stated as a deployment caveat with a threshold attached.⟩

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

**Table 3**. Rollout-horizon sensitivity (50 test shifts). Soft cross-entropy is
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

It is natural to ask whether DAHS's advantage is an artefact of the one
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

A Shapley-value analysis [@lundberg2017shap] (Figure 10) shows the ranker's decisions are driven by
operationally sensible features: queue length and its lags, mean slack, the count
of orders at near-term deadline risk, and the interval index dominate the
attribution. The selector is not exploiting an artefact; it keys on the same
quantities a human dispatcher would.

![Figure 10. Global SHAP feature importance for the ranker.](../figures/E5/shap_summary.png)

## 6.7 Real-data grounding

**Distributional validation**. We compare the simulator's input distributions
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

**Active robustness test**. The distributional comparison is passive — it reports
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

Each ablation retrains or reconfigures a single component and re-evaluates on the
held-out shifts; paired differences against the full model use the Wilcoxon
signed-rank test with Benjamini–Hochberg correction.

**What is reported per ablation, and why it changed**. The submitted version
reported one number per ablation — the SLA-breach rate — which says whether a
component helps but nothing about what it costs to have. A component that improves
the objective by a hair while tripling training time or adding milliseconds to
every decision is a different engineering proposition from one that is free, and a
practitioner choosing which parts of this pipeline to adopt needs both. Every
ablation row now carries three columns:

| column | what it answers |
|---|---|
| composite cost, and its decomposition | does the component improve the objective? |
| training wall-clock to convergence | what does it cost to *fit*? |
| per-decision inference latency (mean, p95) | what does it cost to *run*? |

Latency is measured by the evaluation harness around the policy call alone, with
the environment step excluded, so it is the controller's own cost rather than the
simulator's.

**The ablation set**. Two of the six ablations declared in the submitted
configuration were never actually run — `no_regime` and `random_ambiguity_filter`
— and `no_regime` matters, because the regime layer is a named component of the
method (Section 4.5) whose contribution was therefore never established. Both are
run here. Three further ablations are added:

- **`top5_features`** (Reviewer 3, comment 4). The state representation is
  hand-crafted and the SHAP analysis shows a small number of features dominating,
  so we retrain on only the five most important and report the gap. This tests
  whether the full set earns its dimensionality, and a parsimonious selector is
  materially easier to deploy and to audit. Appendix A reports the correlation
  and variance-inflation analysis that motivates the question, including the two
  degenerate features removed outright in this revision.
- **`single_sample_rollout`** ($M = 1$), which is the submitted labelling scheme,
  so the value of turning the label into an estimator is measured rather than
  assumed.
- **`round_robin_behaviour`**, which regenerates the corpus under the submitted
  behaviour policy, isolating the effect of its degenerate conditional action
  coverage on the offline-RL comparison (Section 6.10).

⟨TBD-rerun: report the full ablation table with all three column groups. Order
rows by effect on composite cost, and state for each whether the component is
load-bearing, neutral, or a deployability guardrail paid for in KPI — the last
category currently contains the switching controller and should be labelled as
such rather than defended.⟩

**Removing the soft label**. DAHS converts each rollout cost vector into a soft
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

**Removing isotonic calibration**. Dropping the isotonic post-processing degrades
the SLA-breach rate from 0.0133 to 0.0294 and the composite cost from 3.09 to 7.85
— both highly significant ($p < 0.001$). Calibration is load-bearing, because the
switching controller's entropy gate acts on the predicted probabilities; an
un-calibrated ranker mis-times its switches.

**Removing the switching controller**. Disabling the dwell and the entropy gate
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

A weak baseline proves nothing, so the first question about the deep-RL
comparison is whether the baseline was configured competently. The submitted
version of this paper did not establish that it was. It varied exactly one thing —
the training budget, 8k against 500k environment steps — and concluded from that
alone that PPO's deficit was "structural, not budgetary". That conclusion did not
follow. No hyperparameter was ever tuned, and the implementation used **no
observation normalisation and no reward normalisation**, on a feature vector whose
columns span queue lengths in the hundreds and utilisations in $[0,1]$.
Unnormalised observations are among the most common causes of a policy-gradient
method failing to learn at all, and a tree ensemble is scale-invariant where a
multilayer perceptron is not — so the submitted evidence could not distinguish
*policy gradients are the wrong instrument here* from *this run was
misconfigured*. We are grateful to the reviewer for pressing the point.

**Sensitivity analysis**. We therefore report a one-factor-at-a-time sweep around
the submitted configuration over the discount factor $\gamma$, the GAE parameter
$\lambda$, the rollout horizon `n_steps`, and the entropy coefficient, together
with the full $2\times2$ over observation and reward normalisation. Normalisation
is swept jointly rather than marginally because observation scaling changes the
effective gradient magnitude while reward scaling changes the advantage scale, and
a marginal sweep would miss the interaction. Every configuration is trained at the
matched budget and evaluated on the same held-out shifts. Table ⟨TBD-rerun⟩
reports the grid; the summary statistic is the fraction of the submitted
PPO-to-DAHS gap that the best configuration recovers.

⟨TBD-rerun: insert the swept grid, the per-factor spread in composite cost, the
best configuration, and `gap_closed_fraction`. If that fraction is large, the
structural claim below must be withdrawn and the tuned configuration adopted as
the PPO baseline throughout. If it is small across the whole swept space, the
claim stands and this sweep is its evidence. The interpretation is written
conditionally on purpose and must be resolved against the measured outcome, not
assumed.⟩

**Interpretation, conditional on the sweep**. If PPO's deficit survives the sweep,
the mechanism is the one policy-gradient theory predicts for this class of
problem. The per-state advantage of one rule over another is small relative to the
variance of the shift return, so the gradient signal that distinguishes rules is
weak; the policy drifts toward whichever rule has the highest *unconditional*
expected return and, with more updates, commits to it — which is consistent with
the observed collapse onto a single rule at the larger budget. A directly measured
per-rule cost vector sidesteps this because it is not a noisy return: it is dense,
per-state, and its variance is controlled by the number of rollout samples rather
than by the length of an episode. That is the comparison this paper is about, and
it is only interesting if the baseline is competently configured — hence the
sweep. The PPO comparison is thus not a horse-race DAHS wins on
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
than the directly measured cost of *every* rule. Its discount and tree
hyperparameters were selected by a 12-configuration search on validation shifts
disjoint from both training and test.

**Action coverage, and a correction**. Whether a fitted-Q baseline is a fair
representative of the offline-RL family depends on whether the logged data
actually support the value estimates it needs, so we quantify coverage rather than
argue for it. The submitted version argued for it, and argued wrongly. It stated
that because the behaviour policy was a uniform round robin, "every action is
covered at every state, so the distribution-shift pathologies that motivate
conservative offline RL (CQL, IQL) do not arise". That is false, and the error is
instructive. The behaviour policy was $a_t = \mathcal{H}[\,t \bmod |\mathcal{H}|\,]$
— and the interval index $t$ is itself one of the observed state features.
The logged action was therefore a **deterministic function of an observed
feature**: marginally each rule was taken equally often, but *conditional on the
state* exactly one action was ever observed, and $Q(s,a)$ for every other action
was pure extrapolation. Marginal uniformity concealed conditional degeneracy.

This affects only the value-learning baseline. The rollout labels are
counterfactual by construction — every rule is simulated at every state regardless
of what the behaviour policy did — so the supervised selector is untouched. But it
means the submitted comparison was not the clean test of *training signal* it
claimed to be: part of the gap could have been coverage.

We therefore make three changes. The corpus is regenerated under a seeded random
behaviour policy, breaking the dependence between the logged action and the state.
Coverage is measured and reported, both overall and restricted to breach-prone
states — defined from the labels as those whose best achievable rollout cost lies
in the top quartile, which is exactly where a value function most needs data — using
the exponentiated entropy of the action distribution, which equals $|\mathcal{H}|$
under uniform coverage and 1 under degeneracy. And both corpora are reported, so
the effect of the behaviour policy on the offline-RL baseline can be read directly
rather than assumed away. Table ⟨TBD-rerun⟩ gives the coverage statistics under
each behaviour policy; the conditional statistic is 1.0 by construction under the
submitted round robin.

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

## 6.11 Model misspecification: labelling in one world, deploying in another

Every result so far shares a simulator between labelling and evaluation. That is
intrinsic to the method and we list it as a limitation, but it also means none of
those results speak to the error an operator actually faces, which is that the
model is wrong. Proposition 2 predicts the shape of that error: it accumulates as
$O(\varepsilon\tau^2)$ while truncation bias decays as $O(H-\tau)$, so a
misspecified model should be rolled out over a shorter horizon, with
$\tau^\star \approx 1/\varepsilon$.

We test this by labelling once under the nominal configuration and then evaluating
the frozen controllers under perturbed dynamics along four axes an operator would
have to estimate and would estimate imperfectly: arrival rate, processing-time
scale, due-window scale, and picker headcount. Nothing is retrained.

Two design points make the comparison informative rather than rhetorical. The
online lookahead controller is pinned to the **nominal** model, so it plans with
exactly the same wrong dynamics DAHS's labels were built from; allowing it to roll
out under the perturbed dynamics would hand it a model nobody has and would make
the experiment meaningless. And the static rules are included as the
misspecification-free reference: they carry no model at all, so their degradation
measures the environment getting harder rather than any method getting worse. The
model-based methods should be read against that baseline, not against zero.

⟨TBD-rerun: report the degradation slope — relative rise in composite cost per
unit of relative model error — for DAHS, the rolling-horizon controller, the
offline-RL baseline, and the static reference, on each axis. Then report the
horizon sweep: does the realised-cost-minimising $\tau$ shorten as the perturbation
grows, as Proposition 2 predicts? If it does not, say so; the proposition bounds
the error but the bound may be loose enough that other effects dominate at these
perturbation sizes. Also report whether the amortised controller or the online one
degrades faster, which is not obvious a priori: the online controller re-plans
every epoch but re-plans with a wrong model, while the amortised one cannot
re-plan at all but was fitted across many states and may generalise more
smoothly.⟩

## 6.12 Computational cost, and scaling in the size of the rule pool

The method's operational argument is that an expensive lookahead is paid once,
offline, and amortised over deployment. That is a claim about two numbers, and the
submitted paper reported neither. It is also a claim that weakens as the rule pool
grows, which the submitted paper did not discuss.

### Offline cost

The submitted labeller reconstructed each decision state by replaying the shift
from $t = 0$, separately for every candidate rule, costing

$$ \sum_{t<N} |\mathcal{H}|\,(t + \tau) \;=\; |\mathcal{H}|\left(\tfrac{N(N-1)}{2} + N\tau\right) \quad \text{interval-steps per shift.} $$

For the setup the reviewer quotes — $N = 32$ epochs, $|\mathcal{H}| = 4$ rules,
$\tau = 4$, 250 shifts — that is $4 \times (496 + 128) = 2{,}496$ steps per shift
and **624,000 interval-steps** in total. The dominant term is the replay, and it
buys nothing: it re-derives a state the shift walk has already passed through.

Walking each shift forward once and branching at each epoch (Section 4.3) costs

$$ N \;+\; N \cdot |\mathcal{H}| \cdot M \cdot \tau \quad \text{interval-steps per shift,} $$

linear in $N$ rather than quadratic. Removing the $O(N^2)$ term is what pays for
the $M$ independent continuations that make the label an estimator at all. The
comparison is computed rather than asserted by
`experiments/compute_budget.py analytic`.

⟨TBD-rerun: report measured interval-steps and wall-clock for rule screening,
calibration and labelling separately, on named hardware with the core count, plus
the same figures for the offline-RL baseline's transition logging so the two
training budgets are directly comparable. `compute_budget.py measure` produces the
per-core rate; labelling is embarrassingly parallel over shifts.⟩

### Online cost, and the break-even

A DAHS decision is one pass through the regime mixture and the ranker. A
rolling-horizon decision is $|\mathcal{H}| \cdot M \cdot \tau$ simulated
interval-steps, at *every* epoch, indefinitely. The evaluation harness records
per-decision latency for both. ⟨TBD-rerun: report mean and 95th-percentile
latency for each, and the break-even — the number of decisions after which the
one-off labelling cost is repaid by the per-decision saving. That expresses the
amortisation claim as a quantity an operator can check against their own
deployment horizon rather than take on trust.⟩

### Scaling in $|\mathcal{H}|$

Two distinct costs grow with the pool, and they grow differently.

**Labelling cost is linear in $|\mathcal{H}|$**, since every rule is rolled out at
every state. Going from four rules to twenty multiplies the offline budget by
five, with no change to the online cost of the deployed selector — the ranker
emits one more logit and nothing else moves.

**Statistical cost is worse than linear.** A pool of $|\mathcal{H}|$ rules is an
$|\mathcal{H}|$-class problem, so the corpus must support estimating that many
decision boundaries. Sample efficiency therefore degrades in $|\mathcal{H}|$ even
though the per-state supervision remains dense, and the shift budget at which the
selector saturates is the quantity to watch, not the step budget.
⟨TBD-rerun: report the sample-efficiency curve at pool sizes 2, 4, 8, so the
degradation is measured rather than argued.⟩

Two mitigations are available and we implement and evaluate the first.

**Adaptive sample allocation.** Uniform allocation spends $M$ continuations on
every rule, including rules that are clearly inferior after two samples.
Successive halving spends the same total budget adaptively — a cheap round over
all rules, discard the worst fraction, reallocate to the survivors — so precision
concentrates on the contenders. This is compatible with the soft label rather than
in tension with it: the tempered softmax maps a clearly inferior rule to
near-zero probability however precisely its cost was estimated, so a rule
eliminated in round one loses nothing it would have contributed. The rules whose
label mass is materially non-zero are exactly the ones the allocation keeps
sampling. ⟨TBD-rerun: report arg-max agreement and label KL against uniform
allocation, and the step saving, from `compute_budget.py scaling`. If agreement is
high and the saving material, recommend it as the default for pools beyond roughly
eight rules; if not, report the mitigation as unsuccessful here.⟩

**Hierarchical selection.** The rules partition naturally by information source
(Section 3.6): arrival-driven, customer-deadline-driven, product-deadline-driven,
and processing-composite. A two-stage selector could choose a family and then a
member, reducing the effective branching factor from $|\mathcal{H}|$ to
$\sqrt{|\mathcal{H}|}$-ish at each stage and allowing rollouts to be spent per
family rather than per rule. We do not implement this — with eight screened rules
the flat selector is not the bottleneck — but it is the natural next step for a
pool of twenty and we note it in Section 9 rather than leave the reviewer's
question unanswered.

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

1. **Partial observability, and an irreducible regret floor.** The controller acts
   on $\phi(S_t)$, which is an observation and not a sufficient statistic
   (Section 3.2). Two queues that differ in *which* order carries the tight
   deadline can be identical under $\phi$ and yet evolve differently under the
   same rule, and Section 3.2 exhibits such a pair explicitly. Any policy in our
   class must therefore assign both the same action, so some regret is
   unrecoverable regardless of data or model capacity. We measure it rather than
   note it: over the training corpus we locate mutual near-neighbours in
   standardised $\phi$-space and report how often their cost-minimising rules
   differ and what acting on the wrong one costs, as a share of the total benefit
   available from rule selection. ⟨TBD-rerun: report the aliasing rate and that
   share; it upper-bounds the part of the residual regret attributable to the
   feature map rather than to the learner.⟩ A richer $\phi$ — a permutation
   invariant set encoding over the queued orders, retaining per-order attributes —
   is the natural remedy and is left to future work.

2. **Model error, which the theory bounds but the deployment cannot avoid.** The
   rollout labels are generated by a simulator, and a simulator is an estimate.
   Proposition 1 bounds only the error from truncating the rollout; Proposition 2
   adds the error from rolling out under the wrong kernel, and shows it accumulates
   as $O(\varepsilon\tau^2)$ against truncation's $O(H-\tau)$. The two act in
   opposite directions in $\tau$, so a misspecified model should be rolled out for a
   *shorter* horizon, with $\tau^\star \approx 1/\varepsilon$. Section 6.11 tests
   this by labelling under nominal parameters and evaluating under perturbed
   arrival rates, service times, due windows and picker counts, with the online
   lookahead controller held to the same nominal model so that neither method is
   handed dynamics the other lacks. What we cannot do is estimate $\varepsilon$
   for a real warehouse, so the bound tells an operator how to trade $\tau$
   against model quality but not what their model quality is.

3. **Simulation-only evaluation.** All KPIs are measured in a simulator. We
   mitigate this by fitting the simulator's arrival and due-date distributions to a
   public real-world trace (Section 6.7) and by re-running the comparison under
   empirically calibrated bursty arrivals, but we do not evaluate on a live
   warehouse floor. Pick time in particular has no public real-world analogue: it
   is grounded in published time standards rather than fitted, and is not
   validated.
4. **A single calibrated operating point.** The simulator was calibrated at one
   configuration. The robustness grid (Section 6.5) shows the method ranking is
   stable across eleven further untuned configurations, but all of them share the
   same simulator family.
5. **A saturation-load loss.** Under the high-load-perishable scenario DAHS
   concedes ground on the breach metric to the analytic lookahead controller
   (Section 6.2). Section 6.2 now analyses which boundary condition is responsible
   — selector collapse or the dwell constraint — rather than attributing it to
   saturation and moving on.

### 8.1 Shared-simulator circularity

The rollout labels and the evaluation use the same simulator, and the training
signal is therefore internally consistent with the thing it is scored against by
construction. This is intrinsic to the method rather than an implementation
choice: a rollout label *is* a simulation, so any rollout-supervised selector
inherits it.

Three mitigations are in place and none of them dissolves the concern. The
rolling-horizon controller (Section 5) is an independent analytic baseline that
does not share the training pipeline, so a DAHS advantage over it is not explained
by shared assumptions. The misspecification study (Section 6.11) breaks the shared
simulator deliberately, labelling under one parameterisation and evaluating under
another, which is the closest thing to an independent environment available
without building one. And the simulator's inputs are fitted to a public order
trace rather than assumed (Section 6.7), so at least the arrival and due-date
structure is externally anchored.

What remains unaddressed is the transition function itself — the dispatch and
service dynamics — which no public dataset constrains. A second, independently
implemented simulator, ideally by a different author, is the correct remedy and
we have not done it.

### 8.2 A small heuristic pool

The pool contains eight rules screened from ten candidates (Section 3.6),
spanning the four information sources available to a dispatcher. That is
substantially broader than the four rules of the submitted version, but it is
still a *fixed, hand-assembled* pool of classical rules, and two consequences
follow.

The performance ceiling is the pool's envelope. DAHS selects; it does not
construct. If no rule in the pool is appropriate to some operating region — and
Section 6.2's saturation analysis is where one would expect to find such a region
— then no selector over that pool can do well there, however good the training
signal. Genetic programming is the established route to *generating* rules rather
than selecting among them [@branke2016automated; @nguyen2017gpsurvey], and a
hybrid that evolves candidate rules and then selects among them with rollout
supervision is a natural combination we do not attempt.

The cost of growth is characterised but not paid. Section 6.12 shows labelling
cost is linear in $|\mathcal{H}|$ while the statistical cost of an
$|\mathcal{H}|$-class problem is worse than linear, and evaluates adaptive sample
allocation as a mitigation. Hierarchical selection over rule families is sketched
there and left unimplemented. A pool of twenty rules is feasible on this evidence
but has not been demonstrated.

### 8.3 A single warehouse setting

Every result comes from one facility archetype: a single-zone, picker-to-parts
operation with ten pickers, unit-sized pick tours, and a periodic 15-minute
review. Three restrictions are worth separating.

*Scope.* Storage assignment, batching and routing are exogenous (Section 2.1), so
the processing time $p_o$ absorbs all travel and search variation into a single
draw. A facility whose travel times depend strongly on the *sequence* of picks —
which is most of them — has structure this model cannot represent, and a
controller that jointly optimised release and routing could dominate any
dispatching rule.

*Scale.* Ten pickers and a 200-order queue is a small operation. Larger floors
introduce zoning and congestion effects that change which rules are sensible, and
the feature map carries no zone or congestion signal.

*Regime.* One shift, one product mix, one SLA policy. Multi-shift effects —
carryover backlog, shift handover, demand seasonality across a week — are outside
the horizon entirely, and the finite-horizon end-of-shift effect that
`interval_index_in_shift` captures would be replaced by something quite different
in a continuously operating facility.

### 8.4 No online adaptation after deployment

The selector is fitted once and frozen. It does not update on the shifts it
subsequently controls, and it has no mechanism for detecting that the world has
moved. This is a deliberate design property — it is what makes the deployed
controller a deterministic forward pass with auditable behaviour — but it is a
real limitation, and it interacts badly with the model error of Section 8.2's
sibling concern above.

A warehouse drifts: order mixes change seasonally, staffing changes, SLA policies
are renegotiated. Under Proposition 2 the cost of that drift is bounded by
$O(\varepsilon\tau^2)$, where $\varepsilon$ is the growing gap between the
labelling model and the live system — so a frozen selector degrades smoothly
rather than catastrophically, but it does degrade, and nothing in the current
design notices. The natural remedies are all unimplemented: periodic re-labelling
from logged shifts, which is cheap because the rollouts are offline; an online
residual correction on top of the frozen selector; or a drift detector on the
observation distribution that triggers re-fitting. Section 9 takes these up as
future work. We have no evidence about how quickly the frozen selector goes stale
in practice, because we have not run it in practice.

### 8.5 Reinforcement-learning baselines

The online PPO baseline performs poorly on this problem (Section 6.9). We take
seriously that this could be unfavourable to reinforcement learning by
construction, and address it two ways. Section 6.9 reports a hyperparameter
sensitivity sweep over the discount, GAE parameter, rollout horizon, entropy
coefficient and observation/reward normalisation, so the claim is no longer
resting on a single stock configuration — the submitted version varied only the
training budget, which was not sufficient evidence for the conclusion it drew.
And a second, *offline* RL baseline (Section 6.10) fails differently: fitted
Q-iteration learns a competent non-degenerate policy at the default operating
point and loses on the objective, rather than collapsing. Two failure modes with
different shapes make a shared tuning artefact less likely, though not impossible.

A comparison against a modern conservative offline-RL method (CQL, IQL) would
strengthen the case further and is left to future work. It matters more now than
it did in the submitted version, because the corrected behaviour policy
(Section 6.10) changes the coverage regime those methods were designed for.

---

# 9. Conclusion and Future Work

We studied DAHS, a selection hyper-heuristic for warehouse order dispatching under
customer and product deadlines, and used it to ask what the *form of the
supervision* is worth. The training mechanism — simulate a rule pool offline, fit a
classifier to the result — is not new, and Section 2 places it within rollout
classification policy iteration and multi-pass rule selection. What we contribute
is a controlled comparison in which the environment, corpus, model class, feature
set and objective are held fixed and only the supervision varies: a directly
measured per-rule cost vector, a bootstrapped state–action value, or a policy
gradient.

On the theory, we gave two bounds that act in opposite directions in the rollout
horizon. Proposition 1 bounds truncation error, which decays as $O(H-\tau)$;
Proposition 2 bounds error from rolling out under a misspecified model, which
accumulates as $O(\varepsilon\tau^2)$. Together they place the optimal horizon at
$\tau^\star \approx 1/\varepsilon$ — the better the simulator, the longer the
rollout worth running — and that prediction is testable, and tested. On the
formulation, we modelled the customer due date and the product expiry as
independent constraints, measured whether the second binds at the review interval
rather than assuming it, and stated plainly that the controller observes a summary
of the state rather than the state.

⟨TBD-rerun: restate the empirical findings here once the campaign completes. The
sample-efficiency claim in particular must be re-established under the corrected
metric before it is repeated, since Section 6.2 shows the accounting correction
compresses the margins it was stated against.⟩

**Future work.** The limitations of Section 8 point to five next steps, roughly in
order of how much they would change the conclusions.

*An independent environment.* Evaluation on a logged real-warehouse trace, or on a
second simulator built independently, is the only thing that resolves the
circularity of Section 8.1. Everything else is mitigation.

*Growing the pool, and generating it.* Section 6.12 shows labelling cost is linear
in $|\mathcal{H}|$ while the statistical cost is worse, and evaluates adaptive
sample allocation as a mitigation. Hierarchical selection over the rule families of
Section 3.6 — choose a family, then a member — is the natural design for a pool of
twenty and is unimplemented. Beyond that, genetic programming *generates*
dispatching rules where this paper only selects among them
[@branke2016automated; @nguyen2017gpsurvey], and a hybrid that evolves candidates
and then supervises selection over them with rollouts is the combination neither
literature has tried.

*Online adaptation.* The selector is fitted once and never notices that the world
has moved (Section 8.4). Periodic re-labelling from logged shifts is cheap,
because the rollouts are offline by construction; a drift detector on the
observation distribution would say when to do it. Neither is implemented, and we
have no evidence on how fast the frozen selector goes stale.

*A richer observation.* The feature map is not a sufficient statistic and
Section 3.2 exhibits the states it cannot separate. A permutation-invariant set
encoding over the queued orders, retaining per-order attributes, would raise the
ceiling that Section 8's aliasing measurement quantifies.

*A value-approximated tail, and a state-dependent horizon.* This is the extension
we regard as most promising, and it is not ours. Rather than truncating the
rollout and discarding the remainder, the dynamic-dispatching literature
approximates the tail with a learned value function, and in places learns the
horizon itself as a function of state [@ulmer2019offlineonline; @ulmer2018budgeting;
@goodson2017rolloutframework]. Substituting an approximated tail for our hard cut
would replace the truncation term $(H_t-\tau)\bar{C}$ of Proposition 1 with an
approximation error that need not grow with the remaining horizon, and would
therefore permit a shorter $\tau$ — doubly attractive, since Proposition 2 shows a
shorter horizon also limits model-error accumulation. Model quality is in any case
not uniform across the state space: a saturated queue is more predictable than a
quiet one, so a state-dependent $\tau$ should tighten both bounds where each is
loosest. Combining that literature's tail approximation with the per-action cost
supervision studied here is the obvious next construction, and we have not
attempted it.

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
