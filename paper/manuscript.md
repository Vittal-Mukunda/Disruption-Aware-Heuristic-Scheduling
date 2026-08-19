---
title: "Offline Rollout Distillation for Warehouse Order Dispatching: A Controlled Comparison of Training Signals"
author:
  - name: Vittal Mukunda
    email: vittalmukunda.im24@rvce.edu.in
    affiliation: Department of Industrial Engineering and Management, R. V. College of Engineering, Bengaluru, India
    corresponding: true
  - name: Atharva Somani
    email: atharvasomani.im24@rvce.edu.in
    affiliation: Department of Industrial Engineering and Management, R. V. College of Engineering, Bengaluru, India
  - name: Pranjal Malaiya
    email: pranjalmalaiya.im24@rvce.edu.in
    affiliation: Department of Industrial Engineering and Management, R. V. College of Engineering, Bengaluru, India
date: 2026
bibliography: references.bib
keywords:
  - dynamic dispatching
  - selection hyper-heuristics
  - rollout
  - approximate policy iteration
  - warehouse operations
  - perishable inventory
abstract: |
  Warehouse order dispatching under customer due dates and product expiry is usually left to priority rules, but no single rule is best across a shift. A selection hyper-heuristic can choose the rule from the current state. Training that selector by simulating candidates offline and fitting a classifier is not new --- it is multi-pass rule selection and rollout classification policy iteration --- and we do not claim a new mechanism. We ask what the form of the supervision is worth, holding environment, corpus, model class and objective fixed.

  The problem is a partially observed sequential decision process. Each order has two independent clocks, both priced, together with tardiness and unserved demand, in a composite cost $J$. Three signals train the same selector: a Monte Carlo truncated rollout of every rule; fitted Q-iteration on the same logs; and a policy gradient.

  On 50 held-out default shifts the rollout-trained selector (DAHS) has mean $J=381$, beating Always-COVERT ($454$; $49/50$ shifts), fitted Q ($397$), and an untuned PPO policy ($611$; $450$ after observation and reward normalisation). Always-EEDD has mean $J=696$ but DAHS is strictly cheaper on only $21$ of $50$ shifts (7 losses, 22 exact ties); the mean is tail-driven. Online lookahead remains cheaper ($356$--$363$) at $176\times$ the latency ($3.7$ ms vs $645$ ms). Labels from $M=1$ through $M=40$ sit in a $0.7\%$ band on $J$. A one-step label is a null on $J$ under the confirmatory paired interval (which includes 0) and a null on service-failure rate; Wilcoxon signed-rank rejects equality on $J$. Distillation amortises a slightly worse scoring rule into a millisecond decision; it does not recover the teacher.
---

# 1. Introduction

Order dispatching on a warehouse floor is a sequential decision problem under
uncertainty: orders arrive stochastically, each carries a due date and possibly a
product expiry, and a small pool of pickers must be assigned work so as to
minimise late and spoiled shipments. In practice the decision is delegated to a
*dispatching rule* --- first-in-first-out, earliest-deadline-first, weighted
shortest-processing-time, and the like --- because rules are transparent, fast,
and require no training. The well-known difficulty is that no single rule
dominates: the rule that minimises lateness under light load is not the rule that
does so when the queue is saturated or when a burst of perishable orders arrives.
A controller that *selects* the rule appropriate to the current state --- a
*selection hyper-heuristic* [@drake2020hyperheuristics; @dokeroglu2024hyperheuristics]
--- can in principle capture the envelope of the pool without abandoning the
operational advantages of rules.

The open question is how to *learn* the selector. The dominant modern answer is
deep reinforcement learning [@mahmoudinazlou2025drl; @zhang2024lstmppo]. DRL is
attractive but sample-hungry, and it is unstable on problems where the per-state
advantage of one action over another is small relative to the return variance ---
exactly the regime of rule selection, where every rule is a reasonable policy and
the differences are at the margin. A second answer is imitation of an expert
dispatcher [@hanjung2025imitation]. Both families face a structural limitation:
DRL never sees the counterfactual cost of the rules it did not take, and imitation
needs an expert. The training signal we study has neither problem: it measures
the counterfactual cost of every rule directly, and needs no expert.

The signal is a rollout [@bertsekas2020rollout]. Fix a state, run each candidate
rule forward for a short horizon, and record the cost it incurs. Rollouts can be
run online at each decision, which is the classical control use and is too slow
for a warehouse controller, or offline once over a corpus of states. For each
state in a corpus of simulated shifts we roll out every rule, obtain a per-rule
cost vector, and fit a supervised ranker to it. The expensive lookahead is
thereby *amortised* into a cheap function approximator: at deployment the ranker
is a single forward pass, and the rollouts live entirely in the training set. We
retain the cost margin between rules through a soft, tempered-softmax label by
default; an ablation (Section 6.8) shows the soft form is not essential.

This paper makes three contributions. None of them is the training mechanism,
which is not new: simulating a rule pool offline and fitting a classifier to the
result is related to rollout classification policy iteration in the
reinforcement-learning literature and to multi-pass rule selection in the
scheduling literature, and Section 2 places the method inside both. The
construction here is a single supervised fit to truncated-rollout labels, not
an iterated policy-iteration loop, so we do not claim to have implemented RCPI.

1. **A controlled comparison of training signals.** Holding the environment, the
   shift corpus, the feature set and the objective fixed, we vary how the
   supervision is constructed: a directly measured per-rule cost vector, a
   bootstrapped state--action value fitted from the same logged transitions
   (same gradient-boosted-tree class as the ranker), and a policy gradient
   (a neural PPO policy; not the same approximator class). Section 6.10 reports
   the comparison, with the action-coverage diagnostics that determine whether
   it is clean and the hyperparameter sensitivity analysis (Section 6.9) that
   distinguishes a structural result from a tuning artefact.
2. **A warehouse formulation with two deadline clocks.** Customer due date and
   product expiry are modelled as independent constraints, both entering the
   objective, and Section 3.5 *measures* whether the second binds at a 15-minute
   review interval rather than assuming it. We also state plainly that the
   controller observes a summary $\phi(S_t)$ rather than the state, exhibit two
   queues the summary cannot separate, and treat the problem as partially observed
   (Section 3.2). Truncation and model-error bounds are stated in Section 4.4 as
   sketches of the labelling object; they are not contributions the campaign
   confirms.
3. **A sample-efficiency result.** Because each training state carries a directly
   measured per-rule target rather than a return, the selector saturates its
   learnable structure by 50 simulated shifts on the default operating point. We
   report that curve against the offline-RL baseline given the same data
   (Section 6.3).

The empirical claim that survives the campaign is narrower than the one we set
out with. On the default operating point DAHS beats every static dispatching rule
on composite cost, with Always-COVERT --- not Always-EEDD --- as the static to
beat. It does not beat the online teachers. The expensive labelling choices
($\tau=4$, $M=20$) do not earn measurable deployed accuracy over $\tau=1$ and
$M=1$, including a completed $M=40$ cell. What the distillation buys is latency:
3.7 ms per decision against 645 ms for the rolling-horizon teacher. Sections 7
and 9 are written around that result, not around recovery of the lookahead.

## 1.1 Terminology and notation

The submitted version used a compact vocabulary without defining it, and terms
such as "corpus of simulated shifts", "held-out shifts", "SLA-breach rate" and
"snapshot-trained ranker" appeared unexplained. Every term used
in the paper is defined here, on first use in the text, or both.

| Term | Meaning |
|---|---|
| **shift** | One 8-hour working period, the unit of simulation. Divided into $N = 32$ review intervals of $L = 15$ minutes. |
| **decision epoch** | The boundary of a review interval, where the controller acts. There are $N$ per shift. |
| **dispatching rule** | A function that orders the waiting queue. Pickers are then assigned down that order. FIFO, EDD and the rest are dispatching rules. |
| **selection hyper-heuristic** | A controller that chooses *which dispatching rule to apply*, as a function of the current state, rather than choosing an assignment directly. |
| **corpus of simulated shifts** | A set of shifts generated from distinct random seeds, used as data. Split into three disjoint blocks: **training** (fits the selector), **calibration** (fits rule parameters such as ATC's look-ahead scale), and **test**. |
| **held-out shifts** | The test block. No DAHS fitting decision (ranker grid, isotonic split, ATC/COVERT $k$, regime $K$) uses a test shift. PPO hyperparameter sensitivity in Section 6.9 was evaluated on this block; that is a limitation of the PPO comparison, not of the DAHS protocol. |
| **SLA** | Service-level agreement --- the contractual delivery commitment. An order's SLA due time $d_o$ is when it must ship. |
| **SLA-breach rate** | The fraction of orders shipped after $d_o$. Two denominators are possible and the choice matters (Section 3.3): over *arrived* orders, or over *completed* orders only. We always state which. |
| **service-failure rate** | Our primary reported KPI component: the fraction of *arrived* orders that either ship late or spoil, whether or not they were ever dispatched. |
| **rollout** | Simulating forward from a state under a fixed rule, to measure what that rule costs. |
| **truncated rollout** | A rollout stopped after $\tau$ intervals instead of running to the end of the shift. |
| **continuation** | One independently sampled future used for a rollout. The label averages over $M$ of them. |
| **ranker** | The fitted classifier that maps an observation to a distribution over rules. A gradient-boosted decision-tree ensemble here. |
| **snapshot-trained ranker** | The ranker fitted to labels from a *one-interval* rollout, i.e. $\tau = 1$. Used as an ablation to isolate the value of looking further ahead. |
| **soft label** | The training target expressed as a distribution over rules rather than a single winner, obtained from the cost vector by a tempered softmax. |
| **regime** | A cluster of operating conditions, fitted by a Gaussian mixture over training observations. Membership probabilities are appended to the observation. |
| **switching controller** | The deployment wrapper that enforces a minimum **dwell** (hold a rule for $T_{\min}$ epochs) and an **entropy gate** (permit an early switch when the ranker is confident). |
| **ablation** | Removing or altering one component and re-measuring, to establish what that component contributes. |
| **DAHS** | Disruption-Aware Heuristic Scheduling, the selection hyper-heuristic studied here. |

**Notation.**

| Symbol | Meaning |
|---|---|
| $t$, $N$, $L$, $T$ | epoch index; epochs per shift (32); interval length (15 min); shift length (480 min) |
| $S_t$ | the true state: queue with full per-order attributes, picker availability, clock |
| $\phi(\cdot)$, $x_t$ | the feature map, and the observation $x_t = \phi(S_t)$ the controller actually sees |
| $u_t$ | the decision at epoch $t$ --- a rule from the pool |
| $W_{t+1}$ | exogenous information: orders arriving in $(t, t+1]$ and their attributes |
| $S^M(\cdot)$ | the transition function (Section 3.4) |
| $\mathcal{H}$, $h$ | the rule pool and a member of it |
| $H_t$ | intervals remaining in the shift at epoch $t$ (distinct from $\mathcal{H}$) |
| $a_o, p_o, d_o, x_o, w_o$ | order $o$'s arrival, processing time, customer deadline, product expiry, economic weight |
| $f_o$ | completion time of order $o$ |
| $J$; $W_b, W_t, W_s, W_h$ | the composite objective and its weights: breach, tardiness, spoilage, holding |
| $\tau$, $M$, $\beta$ | rollout horizon; continuations per rollout; dimensionless softmax-temperature multiplier ($T(s)=\beta\hat\sigma(s)$) |
| $T(s)$ | per-state softmax temperature, in cost units |
| $\varepsilon$, $\bar{C}$ | per-step model error in total variation; a working per-interval cost scale used in the sketches of Section 4.4 (not a proved bound) |

The remainder of the paper is organised as follows. Section 2 reviews related
work. Section 3 defines the dispatching problem and the simulator. Section 4
presents DAHS and the consistency result. Section 5 describes the experimental
protocol. Section 6 reports results. Sections 7--9 discuss, list limitations, and
conclude.
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
and job-shop dispatching with an LSTM-PPO agent [@zhang2024lstmppo]. What distinguishes the warehouse
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
dynamic programming [@powell2022rlso] and are typically truncated to a finite
horizon for tractability [@he2024truncatedrollout].

The construction we use — estimate action values by simulation at a sample of
states, then fit a classifier to represent the improved policy — is the same
family as **Rollout Classification Policy Iteration**, introduced by
@lagoudakis2003rcpi and developed by @fern2006api, @dimitrakakis2008rollout and
@farahmand2015capi. It is not a new training paradigm, and the submitted version
of this paper was wrong to present it as one. In particular, the claims that
rollouts are "normally used online" and that our method "inverts the usual
deployment" were both incorrect: offline rollout generation for supervised policy
learning has existed for over two decades in the reinforcement-learning literature
and, as Section 2.3 records, for longer than that in the scheduling literature.
We withdraw those claims. We also do not run the policy-iteration loop: labels are
generated once under a behaviour policy and the ranker is fitted once, so the
deployed controller is a one-shot imitation of truncated rollouts rather than
iterated RCPI.

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
routing and dispatching problems that are the closest operational neighbour to
Section 3. Section 3 itself uses Powell's sequential-decision notation
[@powell2019unified], not route-based MDPs. The same group's work
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

It is fair to ask what distinguishes this from value-function approximation or
reinforcement learning, since those also learn offline from simulation. The
submitted paper's answer was rhetorical. The substantive answer has three parts, and the first thing to say is that the method
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
Policy Optimization [@schulman2017ppo] under an 8{,}000-epoch budget, with a
hyperparameter sensitivity analysis in
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
   corpus and the objective fixed, and vary how the training signal is
   constructed: a directly measured per-action cost vector, a bootstrapped
   state-action value fitted from the same logged transitions on the same
   gradient-boosted-tree class, and a policy gradient (neural PPO). Section 6.10
   reports that comparison, and Section 6.9 supports it with the hyperparameter
   sensitivity analysis needed to distinguish a structural result from a tuning
   artefact.

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
than a constraint on the problem. The model, the objective and the rule pool are
corrected accordingly, and Section 3.5 tests whether the corrected constraint
actually binds.

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
directly to the implementation, so it never established what decision was being
taken, against what information, or under what objective. This section is the
remedy. We state the problem as a
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
called $x_t$ "the state". That is wrong as terminology and, more importantly, as
mechanism: $\phi$ records *marginal*
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
of on-time completions is strictly smaller than in $\mathcal{Q}^{B}$.

Two conditions are needed for that difference to be *realised*, and both are part
of the construction rather than incidental to it.

*The queue must contend for a picker.* A ranking expresses a preference only when
something has to wait. With the deployed ten pickers and a two-order queue both
orders start immediately whatever the ranking says, every rule produces the
identical trajectory, and the cost gap is zero by construction. The witness is
therefore built at one picker. This is not a weakening: contention is the regime
in which rule selection has any effect at all, so it is the only regime in which
partial observability can cost anything.

*The rule must key on slack and processing time jointly.* Queues $\mathcal{Q}^A$
and $\mathcal{Q}^B$ carry the *same multiset* of slacks, so a rule that sorts on
slack alone — EDD, EEDD, MS, MDD — orders them identically and no witness exists
under it. The composite rules ATC and COVERT rank on slack *and* $p_o$ together,
which is precisely the interaction $\phi$ discards by recording the two marginals
separately. That is the sharper statement of the defect: $\phi$ retains the
marginal distributions of slack and of processing time and destroys their
coupling.

`experiments/observability_analysis.py` searches a grid of picker counts,
processing times and slacks over every rule in the pool, verifies that the feature
vectors coincide to machine precision before comparing anything, and reports the
strongest gap. Under ATC at $\tau = 4$ with one picker it finds
$\phi(S^A) = \phi(S^B)$ to machine precision with costs of $3.79$ against
$-0.01$ — a gap of $3.80$, against a per-order breach weight of $W_b = 3.0$. One
such pair is sufficient: $\phi$ is not a sufficient statistic, and no quantity of
training data recovers from $\phi$ what $\phi$ does not contain.

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

Three features of this objective differ from the submitted version, and each is
consequential.

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
orders admitted by the last review epoch (Section 3.4), $\mathcal{S} \subseteq \mathcal{A}$ for
those dispatched, and $\mathcal{P} \subseteq \mathcal{A}$ for the perishable ones.
The composite cost uses $f_o = T + p_o$ for any order still waiting at the
reference horizon $T$. The reported *rates* do not: an unserved order is late or
spoiled in the KPI only if the relevant clock has already passed at $T$, with no
$+p_o$. That split is deliberate. Adding $p_o$ to the KPI would count an unserved
order whose due date is still in the future as a service failure.

$$ \text{service-failure rate} = \frac{|\{o \in \mathcal{A} : \text{overdue or expired at } T\}|}{|\mathcal{A}|}, \qquad \text{spoilage rate} = \frac{|\{o \in \mathcal{P} : \text{expired at } T\}|}{|\mathcal{P}|}, $$

$$ \text{breach rate}_{\text{arrived}} = \frac{|\{o \in \mathcal{A} : \text{overdue at } T\}|}{|\mathcal{A}|}, \qquad \text{breach rate}_{\text{served}} = \frac{|\{o \in \mathcal{S} : \text{overdue at } T\}|}{|\mathcal{S}|} . $$

The last of these is the metric the submitted paper reported, under the
unqualified name "SLA-breach rate". Its denominator excludes every order the
controller declined to dispatch, and — with $W_h = 0.005$ against $W_b = 3.0$ —
so did the objective. Both are reported here, under names that make the
denominator explicit, so the two versions of the paper remain comparable.
$\mathcal{A}$ includes orders rejected at the door when the queue was at
capacity; they are real demand that went unmet, and excluding them would reopen
the same gap in a different place. $\mathcal{A}$ does **not** include arrivals in
the last open interval $(T-L, T]$: admission runs at review epochs, the last
review is at $t=T-L$, and there is no terminal admit at $T$. Mean $|\mathcal{A}|=767$
against a Poisson mean of $1.65\times 480=792$. Those late arrivals are never
eligible for dispatch and are shared across every method.

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
was the second cause of the anomalous rule performance reported there.

**Parameters and their provenance**. Every input is now either fitted to data,
grounded in a cited source, or declared a design choice; none is left unexplained.

| Input | Value | Provenance |
|---|---|---|
| Shift, review interval, pickers | 8 h, 15 min, 10 | Operating point |
| Queue capacity | 200 orders | Operating point |
| Inter-arrival **shape** | Poisson (exponential inter-arrivals) | Operating point. Olist shape is lognormal; Section 6.7 and Appendix C report the fit and a frozen-ranker replay under empirical bootstrap arrivals |
| Arrival **rate** | 1.65 orders/min nominal | Operating point; swept in Section 6.5 |
| Processing time $p_o$ | Triangular$(2, 5, 12)$ min | **Literature**: three-point time standard for manual picker-to-parts picking [@tompkins2010facilities; @dekoster2007orderpicking] |
| Customer window $d_o - a_o$ | Triangular$(15, 45, 90)$ min | **Fitted** to the Olist purchase-to-estimated-delivery distribution, shape only, rescaled to the shift time base |
| Shelf life $x_o - a_o$ | Triangular$(20, 60, 120)$ min | Design parameter; no public trace carries expiry. Swept in Section 6.4 |
| Perishable fraction | 0.20 | Design parameter; varied by scenario |
| Priority classes and weights | $\{$low, medium, high$\}$ at $(0.50, 0.35, 0.15)$, $w_o \in \{1, 2, 4\}$ | Design parameter |

This inverts the submitted workflow, in which parameters were set and then
*validated* against a public trace. Fitting is the right operation on a trace that
exists, and Section 6.7 now reports the fits, the candidate
families compared by AIC, and — for processing time, where the trace carries no
warehouse pick-time field — the reason no fit is attempted.

## 3.5 Does perishability bind at a 15-minute horizon?

A model can carry a product deadline without that deadline ever changing a
decision. Since we claim the setting is perishability-constrained, we test the
claim directly rather than assert it. The criterion is decision-relevance: in what
fraction of decisions does delaying an order by one review interval alter its
outcome?

At each epoch $t$, a waiting order has exactly two options available to the
dispatcher — served now, completing at $t + p_o$, or deferred, completing no
earlier than $t + L + p_o$. Call the order **expiry-pivotal** at $t$ when those
two straddle its product deadline,

$$ t + p_o \;\le\; x_o \;<\; t + L + p_o , $$

so that one interval of delay is the difference between saleable goods and waste,
and **due-pivotal** when the same holds for $d_o$.

Three conditions were fixed *before* the diagnostic was run, and all three had to
hold for the framing to stand: at least 5% of decisions carrying an expiry-pivotal
order; at least 10% of perishables having their product clock bind before their
customer clock; and the choice of rule moving the spoilage count at at least 10% of
epochs. Had any failed, the correct response was to drop the perishability framing
and the expiry-aware rules, and to report that instead.

**Table 1**. Perishability decision-relevance, 30 calibration shifts, 7,440
decision epochs.

| Quantity | Measured | Threshold |
|---|---:|---:|
| Decisions with an expiry-pivotal order in queue | **35.5%** | ≥ 5% |
| Perishables whose expiry binds before their due date | **27.6%** | ≥ 10% |
| Epochs where the rule choice changes the spoilage count | **91.0%** | ≥ 10% |
| Queued perishables that are expiry-pivotal | 7.3% | — |
| All orders that are expiry-pivotal | 1.4% | — |
| Share of economic weight on expiry-pivotal orders | 1.3% | — |
| Mean spoilage spread across rules, when discriminating | 3.97 orders | — |

All three pre-registered conditions are met, so the constraint is real at this
horizon and the framing stands. Two qualifications belong with that conclusion
rather than after it. The *marginal* rate is small — only 1.4% of individual orders
are expiry-pivotal at any given epoch, carrying 1.3% of economic weight — so
perishability is not the dominant cost driver; the customer clock is, with 95.1% of
epochs carrying a due-pivotal order against 35.5% carrying an expiry-pivotal one.
What makes it decision-relevant is concentration and frequency: the pivotal orders
cluster, so a third of all decisions have at least one in the queue, and at 91.0% of
epochs the choice of rule moves realised spoilage — by about four orders where it
moves it at all. A constraint that changes the outcome at nine epochs in ten is
binding on the controller whatever share of orders it touches.

## 3.6 The rule pool

The pool is not a convenience sample. Candidates span the four information
sources a dispatcher can key on, so that "no single rule dominates" is a
structural property of the design space rather than an empirical accident. Nine
candidates are screened (Section 6.1); the six that survive are marked.

| Rule | Keys on | Source | Retained |
|---|---|---|:--:|
| FIFO | arrival only | zero-information control | — |
| EDD | customer deadline | @jackson1955edd | ✓ |
| **EEDD** | **both deadlines**: $\min(d_o, x_o)$ | this paper (see below) | ✓ |
| MS | customer deadline slack | @conway1967theory | ✓ |
| MDD | customer deadline, degrading to SPT when past due | @baker1982mdd | ✓ |
| FEFO | product deadline only | classical lot-issuing rule | — |
| WSPT | weight and processing time | @smith1956wspt | — |
| ATC | slack × processing, exponential discount | @vepsalainen1987atc | ✓ |
| COVERT | slack × processing, linear truncation | @carroll1965covert | ✓ |

Four points the submitted version left open.

**FEFO is not a due-date rule, and neither rule alone is right**. The submitted
paper described FEFO as "deadline-aware" and implemented it as a sort on $d_o$.
That is EDD, and both now appear under their correct names. But separating them
exposed something the submitted model could not have shown: with two deadlines,
*neither* single-clock rule is the sensible one. EDD ignores expiry entirely.
FEFO ranks on $x_o$, which is $\infty$ for the 80% of orders that are not
perishable, so it dumps every non-perishable order behind every perishable one —
on this order mix that is close to a strawman, and it is why the FEFO mask
(Section 4.3) had to exist at all.

The rule a scheduler would actually write, told that orders carry two deadlines,
sorts ascending on the **effective deadline** $\min(d_o, x_o)$ — whichever clock
binds first. We call it **EEDD**, and it is in the pool because leaving it out
would have made "expiry-awareness buys nothing" an artefact of testing only a
degenerate expiry rule. Section 6.1 reports what it earns.

**ATC is calibrated, not assumed**. WSPT is exactly the $k \to \infty$ limit of
ATC: as the look-ahead scale grows, $\exp(-\text{slack}/(k\bar p)) \to 1$, leaving
the WSPT index $w_o/p_o$. A correctly calibrated ATC therefore cannot be beaten by
WSPT, and the submitted result — WSPT winning 32% of decisions against ATC's 10% —
was a symptom of an unfitted $k$, fixed at 2.0 with no search performed. We
calibrate $k$ twice, on a calibration corpus disjoint from both training and test
shifts: once for **standalone** use, because ATC is itself a reported benchmark
and benchmarking an uncalibrated rule understates it; and once for **portfolio
contribution**, which is the quantity that matters when the rule sits inside a
selector. COVERT's scale is calibrated the same way. Section 6.1 reports both
values and the deployed one.

**Screening is by marginal contribution, not win rate**. A rule earns its slot by
covering states the others handle badly. We therefore report, per rule, both its
win rate and the increase in achievable cost if it were removed from the pool,
with a bootstrap interval on the latter. A rule with a high win rate and zero
marginal contribution is redundant; a rule with a low win rate and positive
marginal contribution is a specialist worth keeping. Win rate alone, which is what
the submitted Section 6.1 reported, cannot distinguish the two. This is also how
we answer what **FIFO** contributes in a due-date-driven setting: it enters as the
zero-information control and the screen reports whether it earns its place.

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
3. For each rule $h$, hold $h$ fixed for the next $\tau$ epochs of continuation
   $m$ (open-loop truncated simulation of that rule — not one improving action
   followed by a different base policy), and record the cost
   $\hat{J}^{\tau}_{h,m}(s_t)$ accrued over that window. Average:
   $$ \hat{J}^{\tau}_h(s_t) \;=\; \frac{1}{M}\sum_{m=1}^{M} \hat{J}^{\tau}_{h,m}(s_t), \qquad \widehat{\mathrm{se}}_h(s_t) \;=\; \frac{\hat{\sigma}_h(s_t)}{\sqrt{M}} . $$
4. Convert the cost vector into a probability distribution by a **tempered
   softmax** at a state-dependent temperature:
   $$ p^{\tau}_h(s_t) = \frac{\exp(-\hat{J}^{\tau}_h(s_t)/T(s_t))}{\sum_{h'} \exp(-\hat{J}^{\tau}_{h'}(s_t)/T(s_t))}, \qquad T(s_t) = \beta\,\hat\sigma(s_t), $$
   where $\hat\sigma(s_t)$ is the standard deviation of that state's own cost
   vector across the pool and $\beta$ is a single dimensionless multiplier fitted
   once on the training corpus.

**Why $M > 1$, and why the submitted labels were not estimates**. In the submitted
implementation every stochastic quantity was pre-sampled when a shift was
constructed, and the labeller replayed that same shift from its start for each
candidate rule. All rules therefore saw *the identical realised future* — the one
belonging to the shift seed. The label recorded which rule was best **in hindsight
on one path**, not which had the lowest expected cost, and the rollout variance
was identically zero. Hindsight-optimal on one path and lowest-in-expectation are
different quantities, and the difference matters for a method whose stated
objective is expected cost. It also means the bias–variance argument of
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

**Why the temperature is per state**. The submitted version divided every cost
vector by one temperature fitted across the whole corpus. That was serviceable
under the submitted objective and is not under the corrected one. Charging orders
that are never served (Section 3.3) makes a state's cost scale with its
outstanding work, so the spread of the per-rule cost vector grows through a shift
— by two orders of magnitude between the first epochs and the last, measured on
the corpus. A single temperature is then set by the expensive late states, and the
early ones, where the rules genuinely differ by little, come out close to uniform;
the median label entropy sits above any target band at every multiplier. Dividing
by the state's own spread asks the question the ranker actually has to answer —
*how much better is the best rule than the others, here* — and leaves the absolute
cost of the state, which the ranker does not predict, out of the label. States
where every rule ties carry no information at any temperature and fall back to the
uniform.

$\beta$ is then selected once, by a one-dimensional search over a grid, so that
the median training-label entropy falls in a target band expressed as a fraction
of $\log|\mathcal{H}|$ rather than in absolute nats — sharp enough to be
informative, soft enough to retain the cost margin. On the regenerated corpus
($|\mathcal{H}|=6$, $M=20$, $\tau=4$, 250 train shifts) the search selected
$\beta = 0.470$ under per-row temperature and achieved median train-row entropy
$0.638$ nats against the band $[0.387, 0.905]$. The same $\beta$ is applied to the
test and calibration corpora, which would otherwise not be on the ranker's scale.

Two corrections are applied consistently in both labelling and deployment: when
the perishable fraction is below 0.05 the FEFO mass is zeroed and the distribution
renormalised, because FEFO cannot rank a queue with no product deadlines; and,
*for the test corpus only*, states whose maximum label probability falls below
$\theta = 2.2/|\mathcal{H}|$ are filtered out as genuinely ambiguous decisions.
The threshold scales with the pool for the same reason the entropy band does.

**The mask is inert in the deployed model, and we leave it in place rather than
delete it.** Screening dropped FEFO (Section 6.1), so there is no expiry-only rule
left for the mask to suppress and it is a no-op on the deployed pool. It is
retained because it is a property of the *labelling and deployment contract*, not
of one pool: any pool containing a rule that ranks on $x_o$ alone needs it, and
removing the code would silently reintroduce the defect if FEFO ever returned. It
should not, however, be counted as a working component of the method as deployed —
the submitted version described it as one, and on the screened pool it is not.
EEDD, which ranks on $\min(d_o, x_o)$, needs no mask: on a queue with no
perishables it degrades continuously to EDD rather than becoming undefined.

**Where the corpora come from**. Shift seeds are drawn once from a single
`SeedSequence` and partitioned into three contiguous, disjoint blocks: training,
calibration and test. The calibration block is new in this revision and exists so
that rule hyperparameters — ATC's and COVERT's look-ahead scales (Section 3.6) —
can be fitted without touching either of the other two; the submitted version had
no such block, which is part of why ATC went uncalibrated. Each shift contributes
one decision state per review interval, so a block of $n$ shifts yields $32n$
states: the test corpus of 50 shifts gives $50 \times 32 = 1600$ states before
filtering. That is the origin of the 1600 test states the submitted version quoted
without explaining, of which 865 then survived its ambiguity filter.
Under the corrected labels 1525 of 1600 test states survive
($\theta = 2.2/|\mathcal{H}| = 0.367$); 33.4\% of training epochs have a
best/second-best gap below one pooled standard error at $M=20$. The filter is
applied to the test corpus only, and never to the training corpus, so no training
state is discarded for being difficult.

The horizon is fixed at $\tau = 4$ for the deployed model; Section 6.4 studies the
choice.

By default DAHS uses the soft label above: a state where FEFO and ATC are
near-tied produces a near-uniform target over those two rules, and the ranker is
trained to reproduce that uncertainty rather than to guess. The arg-max of the
cost vector — a *hard* label — is the natural alternative, and Section 6.8 reports
an ablation that finds the two equivalent. We describe the deployed (soft) model
here and treat the label's form as a design choice rather than as a contribution.

## 4.4 A consistency result for truncated rollouts

The deployed model truncates the rollout at $\tau = 4$ of up to 32 intervals. The
two statements below are sketches of the labelling object, not theorems the
campaign confirms. The implemented window cost is the potential difference
$\Phi(t+\tau L)-\Phi(t)$ of Section 4.3, which can be negative when the queue
drains. Arrivals are Poisson, so the number of orders that can be refused in an
interval is unbounded and a finite deterministic $\bar{C}$ is not proved; we treat
$\bar{C}$ as a working scale.

**Proposition 1 (truncation remainder).**
*Let $\bar{C}$ be a working upper scale on the absolute per-interval contribution
to the window cost. Fix a decision state $s_t$ with $H_t$ intervals remaining
in the shift. For rule $h$, let $J_h(s_t)$ be the full-horizon cost of holding $h$
fixed for the remaining $H_t$ intervals and
$\hat{J}^{\tau}_h(s_t)$ the $\tau$-truncated cost of holding $h$ fixed, $\tau \le H_t$. Then*

*(i) the truncation remainder is bounded in absolute value by the unsimulated tail,*
$$ \big| J_h(s_t) - \hat{J}^{\tau}_h(s_t) \big| \;\le\; (H_t - \tau)\,\bar{C} \;=:\; \Delta_\tau, \qquad \forall h; $$

*(ii) writing the label as a softmax of energies $-J_h/T(s_t)$ with the deployed
temperature $T(s_t)=\beta\hat\sigma(s_t)>0$,*
$$ \mathrm{KL}\!\left(p^{\infty}(s_t)\,\|\,p^{\tau}(s_t)\right) \;\le\; \frac{2\,\Delta_\tau}{T(s_t)}. $$

*Proof sketch.* (i) The unsimulated tail spans $H_t - \tau$ intervals. If each
contributes at most $\bar{C}$ in absolute value, the remainder is at most
$\Delta_\tau$. The holding term can make a realised window cost negative, so the
older lower bound $0 \le J - \hat J$ need not hold pathwise. (ii) The truncated
energies differ from the full-horizon energies by at most $\Delta_\tau/T(s_t)$ in
absolute value. The log-sum-exp normaliser is 1-Lipschitz in the supremum norm of
its arguments, so each log-probability shifts by at most $2\Delta_\tau/T(s_t)$;
summing the KL contribution over the distribution gives the stated bound. Using
the dimensionless multiplier $\beta$ in place of $T(s_t)$ would give a quantity
with the units of cost, which is not a KL bound. $\square$

Although part (ii) is phrased for the tempered-softmax label, part (i) bounds the
rollout *cost vector* itself; the consistency therefore transfers to any label
derived from that vector — including the hard arg-max label whose ablation we
report in Section 6.8. In this sense the proposition is a statement about the
rollout, not about the choice of label.

### Model error, and why Proposition 1 alone is not enough

Proposition 1 bounds the error from stopping the rollout early. It says nothing
about the error from rolling out in the *wrong world*. The rollout is generated by
a simulator, and a simulator is a model: its arrival process, service times and
picker dynamics are estimates. Any misspecification corrupts the labels, and the
corruption compounds along the rollout — potentially flipping the preferred rule. Proposition 1 is silent on
exactly the error most likely to matter in deployment.

We therefore state the second bound. Let $P$ denote the transition kernel of the
real system and $\tilde{P}$ that of the simulator, and suppose the model is
accurate to $\varepsilon$ per step in total variation,

$$ \sup_{s,\,u}\; \big\| P(\cdot \mid s, u) \;-\; \tilde{P}(\cdot \mid s, u) \big\|_{\mathrm{TV}} \;\le\; \varepsilon . $$

The implemented $\hat{J}^{\tau}$ is a function of the simulated trajectory over
the window, so the first interval already depends on the kernel.

**Proposition 2 (model-error accumulation).** *Under the conditions of
Proposition 1, and treating the window cost as an expectation of a bounded
per-interval function of the post-transition state, the $\tau$-truncated rollout
cost computed under $\tilde{P}$ differs from the same quantity under $P$ by at
most*

$$ \Big| \hat{J}^{\tau,\tilde{P}}_h(s_t) - \hat{J}^{\tau,P}_h(s_t) \Big| \;\le\; \bar{C}\,\varepsilon\,\frac{\tau(\tau+1)}{2} \;=:\; \Gamma_\tau^{\varepsilon} . $$

*Proof sketch.* Let $d_k$ and $\tilde{d}_k$ be the state distributions after $k$
transitions under $P$ and $\tilde{P}$ from the common initial state $s_t$, under
the same rule. Transition kernels are non-expansive in total variation, so
$\|d_k - \tilde{d}_k\|_{\mathrm{TV}} \le k\varepsilon$ by induction, with
$\|d_0 - \tilde d_0\| = 0$. The cost of interval $k=1,\ldots,\tau$ is a function
of the state after that transition, so it differs by at most
$\bar{C}\,k\varepsilon$. Summing $k = 1, \dots, \tau$ gives
$\bar{C}\varepsilon\,\tau(\tau+1)/2$. The older sum that started at $k=0$ gave
$\tau(\tau-1)/2$ and was identically zero at $\tau=1$, which cannot be right for a
window cost that already simulates one interval. $\square$

**The three error terms, and what they imply for $\tau$ and $M$.** Collecting
Propositions 1 and 2 with the Monte Carlo error of the estimator in Section 4.3,
the deviation of a computed label from the ideal full-horizon cost under the true
dynamics is bounded by

$$ \underbrace{(H_t - \tau)\,\bar{C}}_{\text{truncation, } \downarrow \text{ in } \tau} \;+\; \underbrace{\bar{C}\,\varepsilon\,\tfrac{\tau(\tau+1)}{2}}_{\text{model error, } \uparrow \text{ in } \tau} \;+\; \underbrace{O_p\!\big(\hat{\sigma}_h / \sqrt{M}\big)}_{\text{estimator, } \downarrow \text{ in } M} . $$

Three things follow. They organise the labelling knobs; they are not confirmed by
deployed $J$.

First, **the envelope in $\tau$ is interior**. Truncation bias falls linearly in
$\tau$ while model error grows quadratically, so the sum of those two terms is
minimised at $\tau^\star \approx 1/\varepsilon$ under the working scale
$\bar{C}$. The better the model, the longer the rollout that the envelope says is
worth running. The submitted paper attributed an interior optimum entirely to
estimator variance, which, as Section 4.3 explains, its
implementation did not actually possess.

Second, **this is a falsifiable prediction**. If we
degrade the model deliberately by evaluating under perturbed dynamics while
labelling under nominal ones, the horizon that minimises realised cost should
shorten as the perturbation grows. Section 6.11 runs that labelling-in-one-world
protocol and reports degradation slopes. It does not retrain $\tau$ as a
function of $\varepsilon$, so the predicted shortening of $\tau^\star$ is
untested. That is a sharper test than reporting robustness only when the
horizon is swept under misspecification; we did not run that sweep.

Third, **the two knobs are separable**. $\tau$ trades truncation against model
error and is bounded by how much we trust the simulator; $M$ controls only the
estimator term and can be raised independently at linear cost. The submitted
design conflated them because with $M = 1$ the estimator term was unbounded and
invisible at the same time.

Both bounds are stated for the rollout *cost vector*, so — as with Proposition 1
part (ii) — they transfer to any label derived from it, including the hard arg-max
of Section 6.8, via $\mathrm{KL}(p^\infty \| p^\tau) \le 2(\Delta_\tau +
\Gamma^\varepsilon_\tau)/T(s_t)$.

**What these bounds are, and are not.** Both are worst-case and both are loose,
and we would rather say so than let a reader discover it. $\bar{C}$ is a working
scale, not a proved bound: Poisson arrivals can overflow a full queue without
limit. Taken as a numerical plug-in with a full queue of 200 orders at the highest
priority weight, $w_o \le 4$, $W_b = 3$ and $W_s = 5$, one obtains
$\bar{C} \ge 6.4 \times 10^{3}$, and hence $\Delta_\tau \approx 1.8 \times
10^{5}$ at $\tau = 4$ — against realised shift costs three orders of magnitude
smaller. Taken as numerical guarantees the sketches are vacuous. Their content
is the *direction and rate* of each term in $\tau$: truncation falls linearly,
model error grows quadratically, and the estimator term falls as $M^{-1/2}$. That
shape is what the envelope uses; the horizon sweep in Section 6.4 does not recover it in deployed $J$. We
use them for nothing else.

**What the deployed horizon implies.** We cannot measure $\varepsilon$ for a real
warehouse, so $\tau^\star \approx 1/\varepsilon$ cannot be evaluated forward. It
can be read backwards, and that is the more useful direction for a practitioner:
choosing $\tau$ is equivalent to asserting a belief about model quality. The
deployed $\tau = 4$ corresponds to $\varepsilon \approx 0.29$ per step in total
variation — a frankly poor model, and a deliberately conservative choice. An
operator who trusts their simulator to $\varepsilon \approx 0.10$ should be
rolling out to $\tau \approx 10$, and the same objective would then reward a
longer horizon than anything we test here.

That inversion also bounds what Section 6.11 can detect. The horizon sweep runs
over $\tau \in \{1,2,3,4\}$, and $\tau^\star$ enters that range only once
$\varepsilon \gtrsim 0.29$. Under mild perturbation the predicted optimum lies
*beyond* the grid, so the sweep would show cost falling monotonically in $\tau$ —
which is consistent with the prediction but does not test it. Only the more
strongly perturbed cells can exhibit $\tau^\star$ actually moving inward. We
report the sweep with that detection floor stated, rather than reading a monotone
curve as either confirmation or refutation.

## 4.5 Regime discovery

Warehouse shifts pass through qualitatively distinct operating regimes — a quiet
opening, a saturated mid-shift, a perishable burst. DAHS makes this explicit. A
Gaussian mixture model is fit to the training-state features; the number of
components is chosen by BIC over $K \in \{2,3,4,5,6,7,8,10,12\}$ with five EM
restarts per $K$. On the regenerated corpus BIC selects $K^\star = 12$, the
**upper endpoint of the grid** (mean pairwise ARI $0.970$, above the $0.85$
stability threshold). Because the selected $K$ sits on the boundary, the grid —
not a turning point in the data — chose the number of regimes; we report that
rather than treating twelve as a discovered structure. The twelve soft
regime-membership posteriors are appended to $\phi(S_t) \in \mathbb{R}^{26}$, so
the ranker sees a $38$-dimensional vector. Regime discovery remains a lightweight
component; Section 6.8 ablates it.

## 4.6 The calibrated ranker

The ranker is a gradient-boosted decision-tree classifier
[@chen2016xgboost] with an $|\mathcal{H}|$-class soft-probability output
($|\mathcal{H}|=6$ after screening). The soft target is
fitted by an inverse-entropy-weighted replication of each training state across the
six classes, which makes the training objective the Kullback–Leibler divergence
between the predicted distribution and the soft label and down-weights states
whose labels are near-uniform (and therefore carry little discriminative signal).
The hard-label variant of Section 6.8 instead uses a standard one-hot
cross-entropy; the rest of the pipeline — feature set, cross-validation,
calibration — is identical. Hyperparameters are selected by 5-fold cross-validation
with folds grouped by shift, so no shift contributes states to both a training and
a validation fold. The reference point for the cross-validated soft cross-entropy
is the uniform label, $\log|\mathcal{H}|$, which moves with the screened pool
rather than being fixed at $\log 4$ as in the submitted version.
The selected configuration is `max_depth`$=6$, `n_estimators`$=200$,
`learning_rate`$=0.03$, with mean grouped-CV soft cross-entropy $0.875$ against
the uniform baseline $\log 6 \approx 1.79$.

Tree ensembles are not probability-calibrated out of the box. DAHS post-processes
the ranker output with isotonic regression fit on a held-out 20% of training
shifts. The acceptance threshold on expected calibration error was pre-registered
at 0.05 and is unchanged; Section 6.6 reports the achieved value and whether it
clears.

## 4.7 The switching controller

At deployment the calibrated ranker emits, each interval, a distribution over the
retained pool $\mathcal{H}$. A thin *switching controller* maps that distribution
to an action. It
(i) applies the same expiry-rule mask used at labelling time, which is a no-op on
the screened pool (Section 4.3); (ii) enforces a minimum
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
method: *low load* (arrival rate $1.0$, no perishables), *balanced* (rate $1.5$), and
*high-load-perishable* (rate $2.2$, perishable probability $0.4$, customer windows
scaled to $(12,36,72)$ minutes — $0.8\times$ the default triangular).
Scenario parameters were fixed before evaluation and are not tuned per method.

**Baselines**. We compare DAHS against the static rules retained by the screen of
Section 3.6; **snapshot_xgb**, an ablation identical to DAHS but with the rollout
horizon collapsed to $\tau = 1$, isolating the value of the horizon; **LinUCB**
[@li2010linucb], a contextual bandit, with features standardised (LinUCB updates
on the test shifts and is therefore not a frozen peer); **PPO**
[@schulman2017ppo] at 8{,}000 training epochs, with the hyperparameter
sensitivity analysis of Section 6.9 (not a matched simulation budget: labelling
uses millions of interval-steps); and **offline_fqi**,
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
the plan, and replans. Labelling uses $M=20$ continuations; the online teachers
use $M=5$ so that a 50-shift evaluation is affordable. The scoring rule is the
same potential difference, but the Monte Carlo budget is not, so a gap between
rolling_mpc and DAHS mixes function-approximation error with estimator noise and
cannot be read as a pure distillation gap. greedy_mpc beating rolling_mpc on
default $J$ is consistent with a noisy $M=5$ arg-min at $\tau=4$. The one-step
controller is retained as its $\tau = 1$ special case.

This baseline is what makes the paper's central claim falsifiable, and it answers
four questions the submitted version could only assert answers to. *How much does
distillation lose?* The gap between rolling_mpc and DAHS at the same $\tau$ is the
price of replacing a lookahead with a forward pass. *What does it buy?* We report
per-decision latency for both, so the amortisation appears as a measured ratio
rather than an argument. *Can the student beat the teacher?* Section 6.2
reports that it does not: both online lookaheads are cheaper, with paired
intervals excluding zero. A fitted selector could in principle regularise
noisy rollouts; that is not what happens here. *And is the
horizon the mechanism?* Comparing rolling_mpc at $\tau = 1$ against $\tau = 4$
separates the value of lookahead depth from the value of learning.

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
10,000-resample bootstrap 95% confidence intervals. **Confirmatory test for
paired composite cost.** A method is declared different from DAHS on $J$ when the
paired 95% bootstrap interval of the mean difference excludes zero. Wilcoxon
signed-rank $p$-values with Benjamini--Hochberg control are reported as a
sensitivity analysis and are not the confirmatory rule. Service-failure rate uses
the same interval rule when it is the claim being tested.

# 6. Results

## 6.1 Rule calibration, screening, and complementarity

If one rule were best everywhere, selection would be pointless. Establishing that
it is not requires three things the submitted version did not provide: calibrated
rules, a screen that distinguishes redundant rules from specialists, and evidence
of complementarity across the **state space** rather than across instances.

This section is complete: it runs on the 30-shift calibration block, which is
disjoint from both training and test, and it does not depend on the trained
selector. Its results therefore stand ahead of the rest of Section 6.

### Calibration

**Table 2**. Look-ahead scale calibration, 30 calibration shifts, $M = 5$
continuations under common random numbers. Standalone cost is the composite
objective with the rule used alone; portfolio marginal is the cost increase when
the rule is removed from the pool at that $k$.

| Rule | $k^\star_\text{standalone}$ | cost at $k^\star$ | $k^\star_\text{portfolio}$ | marginal at $k^\star$ | deployed |
|---|---:|---:|---:|---:|---:|
| ATC | 1.5 | 459.2 | 3.0 | 0.92 | **3.0** |
| COVERT | 4.0 | 404.3 | 4.0 | 3.21 | **4.0** |

The portfolio value is deployed, because the pool is what ships; both are reported
because ATC is also a standalone benchmark. The Always-ATC row of the default
comparison (Section 6.2) uses the portfolio scale $k=3.0$, not the standalone
$k^\star=1.5$. A standalone Always-ATC at $k=1.5$ was not re-evaluated on the 50
test shifts.

**This settles the WSPT/ATC inversion.** ATC's standalone cost is U-shaped in $k$
with a minimum of 459.2 at $k^\star = 1.5$, and rises monotonically thereafter to
1004.2 at $k = 20$ --- a factor of 2.19. Since WSPT is exactly the $k \to \infty$
limit (Section 3.6), that curve *is* the ATC-to-WSPT interpolation, and it says
a fitted ATC beats WSPT by more than two-fold on this problem. The submitted
finding that WSPT won 32% of decisions against ATC's 10% was therefore an artefact
of the unfitted $k = 2.0$, not a property of the rules. The two values also differ
by a factor of two ($k^\star_\text{standalone} = 1.5$ against
$k^\star_\text{portfolio} = 3.0$), which is the concrete case for calibrating both:
the scale that makes ATC best *alone* is not the scale that makes it most useful
*inside a pool*, where its job is to cover states the other rules handle badly.

### Screening

**Table 3**. Pool screening on the same corpus. Screening is by marginal contribution
to composite cost. Marginal contribution is the increase in achievable cost when
the rule is removed, with a percentile bootstrap interval; a rule is retained when
that interval excludes zero.

| Rule | win rate | marginal contribution | 95% CI | retained |
|---|---:|---:|---|:--:|
| **EEDD** | 0.650 | 5.403 | [4.840, 5.991] | yes |
| COVERT | 0.145 | 2.047 | [1.613, 2.527] | yes |
| MS | 0.070 | 0.248 | [0.119, 0.412] | yes |
| ATC | 0.055 | 0.086 | [0.032, 0.155] | yes |
| MDD | 0.011 | 0.039 | [0.014, 0.070] | yes |
| EDD | 0.068 | 0.007 | [0.000, 0.021] | yes |
| FIFO | 0.001 | 0.000 | [0.000, 0.000] | no |
| WSPT | 0.000 | 0.000 | [0.000, 0.000] | no |
| FEFO | 0.000 | 0.000 | [0.000, 0.000] | no |

**FIFO earns nothing.** It is the cost-minimising rule at 0.1% of decisions and
its marginal contribution is identically zero. As the zero-information control in
a due-date-driven setting that is the expected result, and it is the direct answer
to the question of what FIFO was doing in the pool: nothing, and it is dropped.
Note that FIFO's flattering position in the submitted results, reproduced in
Section 6.2, is separately explained by the admission defect of that section,
which penalised every arrival-agnostic rule and never penalised FIFO.

**WSPT earns nothing either, and that is consistent rather than surprising.** A
calibrated ATC dominates it by construction, so once ATC is fitted, WSPT is a
strictly worse member of the same family and contributes nothing at the margin.

**FEFO's failure is not evidence against expiry-awareness.** FEFO ranks on $x_o$,
which is infinite for the 80% of orders that are not perishable, so it sorts every
non-perishable order behind every perishable one. It contributes nothing because it
is a bad rule on this order mix, not because the product clock is uninformative ---
and reading its zero as "expiry-awareness buys nothing" would have been the wrong
inference. EEDD, which reads $\min(d_o, x_o)$, settles it: it wins 65% of decisions
with a marginal contribution of 5.403, two and a half times COVERT's and by far the
largest in the pool. Expiry-awareness matters a great deal on this problem; the
screen was rejecting a poor implementation of it.

**One rule exceeds the pre-registered concentration ceiling.** EEDD's 0.650 win
rate is above the 0.60 ceiling fixed in advance as the point beyond which a pool is
too concentrated for selection to be worthwhile. We report it rather than drop the
rule to get under the gate, and we read it as a genuine caution about the headline:
a pool with one rule winning two-thirds of decisions leaves a selector less room
than the submitted four-rule pool appeared to. Whether that room is enough is what
Section 6.2 measures, and the per-cell oracle gap below bounds it.

### Complementarity

The submitted Figure 1 reported win rate per shift and per interval. That varies
the *instance*, not the state, and so cannot establish that the rules cover
different operating regions --- which is what "complementary" has to mean for a
state-conditioned selector. Figure 1 is replaced by win rate over a grid of the two
state dimensions that govern the decision: queue length and deadline pressure (mean
slack), in quantile bins. A pool is complementary when different rules own
different cells.

**Table 4**. Cell ownership and the oracle gap over the 4x4 grid (960 decision
epochs from the calibration corpus).

| Quantity | Value |
|---|---:|
| Grid cells | 16 |
| Cells owned by EEDD | 15 |
| Cells owned by COVERT | 1 |
| Decisions won by the best single rule (always EEDD) | 65.00% |
| Decisions won by the per-cell oracle | 72.29% |
| **Oracle gap over the best single rule** | **7.29 pp** |

**We report this against ourselves.** One rule owns fifteen of sixteen cells. At
this resolution the pool is not complementary in the way the submitted Figure 1
was taken to show, and a selector that reads only queue length and deadline
pressure could beat "always EEDD" on at most 7.29 percentage points of win rate.
That is a much smaller opening than the submitted four-rule pool appeared to
offer, and it is the correct place to say so --- before Section 6.2 rather than
after it.

Two qualifications matter for reading it, and neither rescues the number so much
as bound it in the other direction.

*The grid oracle is a floor on the available headroom, not a ceiling on DAHS.*
72.29% is what a selector restricted to a coarse 4x4 partition of two state
dimensions could achieve. DAHS reads 26 features (Section 4.2), so a finer
partition of the state space can only raise the oracle. The gap is therefore a
lower bound on the room available to a richer selector, and the honest statement
is that *this projection* of the state space does not by itself justify selection.

*Win rate is not the objective.* A rule can win rarely and still be worth its slot
if the states it wins are expensive ones, which is exactly why Table 3 screens on
marginal contribution rather than on win rate --- and the two orderings differ:
COVERT wins 14.5% of decisions but carries a marginal contribution of 2.047,
while EDD wins 6.8% and carries 0.007. The quantity that decides whether selection
pays is composite cost, and Section 6.2 measures it. We did not compute a
composite-cost oracle on the same 4x4 grid. What we can say from the test set is
that the static champion on $J$ is Always-COVERT, not Always-EEDD, and that DAHS
beats Always-COVERT on 49 of 50 shifts (Section 6.2). Win-rate concentration on
EEDD therefore overstates how little room there is on the objective.

![Figure 1. Rule complementarity over the state space: win rate of each retained
rule across a grid of queue length (quantile bins) against deadline pressure (mean
slack, quantile bins). Complementarity means different rules own different cells.
The submitted Figure 1 plotted win rate per shift and per interval, which varies
the instance rather than the state and cannot establish
this.](../figures/S1_calibration/diversity_state_grid.png)

Figure 1 is the state-space win-rate grid described above.

## 6.2 Main comparison

Every number in this subsection is regenerated. The objective, the metric, the
admission rule and the rule pool all changed (Sections 3.3, 3.4, 3.6), so no
result carried over from the submitted version is a claim about the model this
paper now describes. The submitted table is reproduced below, clearly marked, for
one purpose only: the two corrections diagnosed in this subsection are visible in
it, and the argument that they are corrections rather than tuning is easier to
follow with the symptomatic numbers in view.

Rank the table by composite cost. Table 5 is the submitted scoreboard, retained
only as a diagnosis; Table 6 is the live comparison.

**Table 5 (superseded)**. The submitted results: default scenario, 50 test
shifts, under the *old* objective, the *old* completed-orders-only breach metric
and the *old* four-rule pool. Retained solely as the evidence for the two
diagnoses below. No claim in this paper rests on these figures.

| Method | SLA breach (completed only) | Mean tardiness | Composite cost (old) | Throughput | Picker util. |
|---|---:|---:|---:|---:|---:|
| DAHS (ours) | 0.0133 | 0.525 | 3.09 | 721.6 | 0.936 |
| greedy_mpc | 0.0313 | 1.822 | 9.19 | 671.5 | 0.846 |
| snapshot_xgb ($\tau{=}1$) | 0.0373 | 1.589 | 8.77 | 673.7 | 0.851 |
| ppo_fair (8k) | 0.0385 | 0.257 | 3.92 | 740.9 | 0.970 |
| FIFO | 0.0660 | 0.618 | 7.57 | 750.6 | 0.983 |
| LinUCB | 0.0694 | 5.208 | 23.61 | 624.1 | 0.771 |
| offline_fqi | 0.0718 | 0.531 | 7.46 | 734.8 | 0.960 |
| WSPT | 0.0949 | 10.671 | 43.56 | 574.5 | 0.686 |
| EDD (submitted as "FEFO") | 0.1181 | 0.997 | 12.60 | 723.8 | 0.947 |
| ppo_full (500k) | 0.1181 | 0.997 | 12.60 | 723.8 | 0.947 |
| ATC (uncalibrated, $k = 2$) | 0.1572 | 1.238 | 16.37 | 721.7 | 0.940 |

Two rows deserve their labels. The rule reported as FEFO sorted on the customer
due date and was in fact EDD (Section 3.6). ATC ran at an unfitted $k = 2.0$
(Section 3.6), so its row understates it. The `ppo_full` row coincides exactly
with that rule's row because the 500k-step PPO policy collapsed onto always
selecting it (Section 6.9).

### The corrected accounting, and what it costs the headline

The submitted paper's main table, reproduced above as Table 5, ranked methods
by a breach rate whose denominator was *completed* orders. That leaves an
opening --- a controller can lower the reported rate by declining to touch
difficult orders --- and the submitted table carries the
direct evidence: DAHS completed 721.6 orders on average against basic FIFO's
750.6. The correct accounting counts every overdue order as a failure, whether it
was completed late or abandoned in the queue.

We regard this as the most important correction in the revision, so we state its
consequence before reporting the new numbers rather than after. Recomputing the
corrected metric on the submitted demonstration logs (ten shifts, frozen model)
--- counting every arrived order, served or not --- gave DAHS 15.00% failures
over arrived orders against a 3.10% completed-only breach rate, and FIFO 17.97%
against 11.75%. That arithmetic narrowed the FIFO gap from roughly 3.8 times to
about 1.20 times *on those logs*. Those logs still used the admission defect of
Cause 2 below, which uniquely favoured FIFO. After causal admission, utilisation
is 0.956 for every method including FIFO, and Table 6's FIFO gap is 3.90 times
on composite cost (50/50 shifts). The 1.20 figure is a diagnosis of the submitted
metric, not a target for the regenerated campaign.

### Two anomalies in the submitted results, and their causes

The submitted version of this table contained two results that are hard to
reconcile with scheduling theory. WSPT --- a shortest-processing-time rule, which
should *maximise* the number of orders completed --- recorded the **lowest**
throughput of any method (574.5 against FIFO's 750.6) and a picker utilisation of
0.686 while its queue sat near the 200-order capacity. And FIFO, which uses no
deadline information at all, placed fourth of eleven on composite cost. Both are
artefacts of the environment and the objective, not properties of the rules, and
both are corrected in this revision.

**Cause 1: the objective did not measure what the rules optimise**. WSPT and ATC
rank by $w_o/p_o$, using the priority weights of Section 3.1. The submitted
objective weighted every order equally. Those two rules were therefore being
graded against a criterion they were not designed for --- they were correctly
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
Section 3.4 replaces this with a properly causal periodic-review admission rule ---
only orders that have arrived by $t$ are eligible at $t$ --- which removes the
handicap and, incidentally, removes fifteen minutes of undisclosed look-ahead from
the observed state.

Together these mean the submitted rule comparison was not measuring rule quality.
All results in this section are regenerated under the corrected environment and
objective, with the recalibrated pool of Section 3.6.

**Table 6.** Default scenario, 50 test shifts, corrected objective and causal
admission. Ranked by composite cost, the quantity every learned method
optimises. Arrived $=767$ on every method. Dropped $=0$. Paired 95% intervals
are bootstrap percentile intervals of (method $-$ DAHS) composite cost over the
50 aligned shifts, 10,000 resamples.

| Method | Composite cost | SFR | Spoil | Tardiness | Thru. | Util. | Latency (ms) | Cost vs DAHS | DAHS wins |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| greedy_mpc ($\tau{=}1$, $M{=}5$) | 356.14 | 0.0607 | 0.0372 | 0.725 | 732.5 | 0.956 | 588 | 0.93$\times$ [$-$39.4, $-$12.8] | 19 |
| rolling_mpc ($\tau{=}4$, $M{=}5$) | 362.58 | 0.0642 | 0.0392 | 0.770 | 732.5 | 0.956 | 645 | 0.95$\times$ [$-$28.1, $-$10.8] | 15 |
| **DAHS** | **381.42** | **0.0689** | **0.0395** | **0.775** | **731.9** | **0.956** | **3.66** | --- | --- |
| snapshot_xgb ($\tau{=}1$) | 388.13 | 0.0671 | 0.0337 | 0.734 | 732.0 | 0.956 | 3.30 | 1.02$\times$ [$-$3.6, 18.2] | 32 |
| offline_fqi | 396.80 | 0.0724 | 0.0466 | 0.802 | 732.0 | 0.956 | 3.19 | 1.04$\times$ [2.9, 29.8] | 31 |
| COVERT | 454.36 | 0.0836 | 0.0857 | 0.811 | 732.1 | 0.956 | $<$0.01 | 1.19$\times$ [60.9, 85.0] | 49 |
| LinUCB | 551.55 | 0.0863 | 0.0694 | 0.773 | 731.2 | 0.956 | --- | 1.45$\times$ [94.7, 264.2] | 46 |
| ATC ($k{=}3.0$) | 559.92 | 0.1108 | 0.0945 | 1.167 | 734.3 | 0.956 | $<$0.01 | 1.47$\times$ [150.9, 206.6] | 50 |
| ppo_fair (untuned) | 610.93 | 0.0951 | 0.0916 | 0.876 | 730.5 | 0.956 | 0.17 | 1.60$\times$ [114.0, 380.5] | 50 |
| EEDD | 695.77 | 0.0943 | 0.0357 | 0.784 | 729.8 | 0.956 | $<$0.01 | 1.82$\times$ [141.4, 528.8] | 21 |
| MDD | 733.15 | 0.0999 | 0.1019 | 0.653 | 730.5 | 0.956 | $<$0.01 | 1.92$\times$ [200.9, 540.9] | 49 |
| EDD | 763.06 | 0.1014 | 0.1033 | 0.746 | 729.8 | 0.956 | $<$0.01 | 2.00$\times$ [211.9, 594.5] | 49 |
| MS | 789.82 | 0.1040 | 0.1058 | 0.782 | 729.1 | 0.956 | $<$0.01 | 2.07$\times$ [225.1, 635.7] | 49 |
| WSPT | 1215.70 | 0.0987 | 0.0611 | 5.554 | 743.3 | 0.955 | $<$0.01 | 3.19$\times$ [753.4, 911.9] | 50 |
| FIFO | 1485.97 | 0.1894 | 0.0743 | 2.023 | 730.2 | 0.956 | $<$0.01 | 3.90$\times$ [895.5, 1342.4] | 50 |
| FEFO | 1698.96 | 0.2032 | 0.0001 | 2.821 | 730.3 | 0.956 | $<$0.01 | 4.45$\times$ [1077.5, 1584.8] | 50 |

Wins are shifts on which DAHS is strictly cheaper. Under the confirmatory paired
interval, every static rule, both teachers, offline_fqi, LinUCB and Table 6's
`ppo_fair` row reject equality with DAHS on composite cost. snapshot_xgb does
not: the interval includes zero ($[-3.6, 18.2]$). Wilcoxon+BH on the same 15
comparisons rejects snapshot_xgb ($p_{\mathrm{adj}}=0.029$; 32 wins, 14 losses, 4
ties, zeros discarded). On service-failure rate the snapshot pair is a null
($p_{\mathrm{adj}}=0.34$). We treat $\tau=1$ as not demonstrably different from
deployed DAHS on $J$ under the confirmatory rule, and as a null on SFR.

![Figure 2. Default-scenario composite cost by method, 50 held-out shifts.
Paired against DAHS.](../figures/E2/default_forest_composite_cost.png)

Figure 2 is that forest plot.

Three facts that rewrite the submitted story:

1. **The teacher does not beat one-step lookahead**, and DAHS does not recover
   the teacher. Distillation is an amortisation of a *worse* scoring rule at
   $M=5$, not of an oracle. Per-decision latency is the quantity DAHS wins
   (3.7 ms vs 645 ms for rolling_mpc; Section 6.12).
2. **Always-COVERT, not Always-EEDD, is the static to beat on cost.** Win rate
   on the Section 6.1 grid is the wrong proxy for the objective. Against COVERT,
   DAHS wins 49 of 50 shifts at 1.19 times; against EEDD the mean ratio is 1.82
   times but DAHS is strictly cheaper on only 21 shifts, EEDD on 7, and they tie
   on 22. The EEDD mean is tail-driven (worst shift 5281 against DAHS 2421). DAHS
   also wins the median (44.1 against 57.6).
3. **WSPT now has the highest throughput** (743 vs FIFO 730) and every method
   sits at picker utilisation $\approx 0.956$. The submitted utilisation of
   $0.686$ is gone. Cause 2 was the mechanism.

**Table 7.** Composite cost and service-failure rate across four scenarios, 50
shifts each. Best static is the cheapest always-on rule in that scenario.

| Scenario | DAHS $J$ / SFR | Best static | Teachers ($J$) | FQI $J$ | PPO $J$ |
|---|---|---|---|---:|---:|
| default | 381.42 / 0.0689 | COVERT 454.36 | greedy 356.14; rolling 362.58 | 396.80 | 610.93 |
| high-load-perish | 11427 / 0.559 | **WSPT 11169** | greedy 11464; rolling 11339 | 11468 | 23223 |
| balanced | 17.05 / 0.00384 | EEDD 16.95 | greedy 17.24; rolling 16.65 | 17.82 | 34.98 |
| low load | 9.04 / 0.00337 | COVERT (13-way tie at 9.04) | 9.04 | 9.04 | 9.04 |

DAHS does not dominate across regimes. Under high-load-perishable WSPT is
cheaper (paired cost difference $-258$, interval $[-332,-187]$, BH-reject).
Under balanced, EEDD is cheaper by $0.10$ with a null test (interval
$[-0.31, 0.00]$). Under low load thirteen methods, including DAHS and both
teachers, return identical $J=9.04$. The default-scenario ranking is therefore
not a licence to write that selection beats any single rule in every operating
region.

### Boundary conditions: what the selector actually does under saturation

Calling a lost cell "a saturation effect" is an attribution, not an analysis.
There are two candidate explanations and they have different remedies.

The first is that the selector stops selecting. We measure this as the
**exponentiated entropy of the deployed-rule distribution**, which equals
$|\mathcal{H}|$ when the selector spreads across the pool and 1.0 when it has
collapsed. On 1,600 default epochs the EEDD share is 0.849 and the COVERT share
is 0.138 (switch rate 0.039). Under high-load-perishable those shares invert:
COVERT 0.828, EEDD 0.173. Under balanced and low load the selector is
effectively Always-EEDD (shares 0.995 and 0.999). Collapse onto a specialist is
the mechanism in the light-load cells; a switch into COVERT is the mechanism
under perishable saturation.

The second is that the guardrail binds. The minimum dwell holds a rule for
$T_{\min}$ epochs, and under saturation the queue state changes fastest --- exactly
when a stale rule is most costly. We measure this as the **blocked-switch rate**:
the share of epochs at which the ranker's arg-max differed from the deployed rule
*because the dwell was still active*. A $T_{\min}$ sweep *within*
high-load-perishable finds the cost-minimising dwell at the deployed value
$T_{\min}=2$ (cost 11426.95); $T_{\min}\in\{0,1\}$ is 11427.04 and $T_{\min}=4$ is
11429.34. The guardrail is not the boundary condition in that scenario.

## 6.3 Sample efficiency

Figure 3.

![Figure 3. Sample efficiency. DAHS mean composite cost versus the number of
simulated training shifts, five independent labelling-and-training replicates
except $n=250$ (one replicate, the deployed model). Source
`figures/data_efficiency/data_efficiency_curve.png`.](../figures/data_efficiency/data_efficiency_curve.png)

**Table 8.** DAHS data-efficiency, composite cost. Budgets
$\{25,50,100,150,250\}$; five replicates except $n=250$ (one replicate by
design).

| $n$ shifts | Replicates | Mean $J$ | Mean SFR | Notes |
|---:|---:|---:|---:|---|
| 25 | 5 | 448.32 | 0.0738 | One of five collapsed to Always-EEDD ($J=695.77$) |
| 50 | 5 | 382.61 | 0.0686 | Already at the deployed level |
| 100 | 5 | 383.03 | 0.0690 | |
| 150 | 5 | 381.95 | 0.0685 | |
| 250 | 1 | 381.42 | 0.0689 | Deployed model |

The selector saturates by 50 shifts. The $n=25$ collapse is a real failure mode
--- not every short corpus yields a working ranker --- but it is one replicate in
five, not the typical outcome. Fitted Q-iteration on the same budgets is slower
to saturate (mean $J$: 580, 483, 456, 412, 397 at $n=25,50,100,150,250$;
Section 6.10).

## 6.4 Rollout horizon, and the number of continuations

Proposition 1 predicts that truncation bias shrinks as $\tau$ grows. The
operational test is a sweep over $\tau \in \{1,2,3,4\}$, each arm labelled at
$M=20$ and used to train an otherwise identical ranker, evaluated on the same 50
test shifts. Cross-validated soft cross-entropy on the training labels is  not
the deployment criterion; we report it alongside $J$ because it is what the
ranker actually fits.

**Table 9.** Rollout-horizon sweep. Cost intervals are 95% bootstrap intervals of
the 50-shift mean. $\tau=1$ is byte-identical to snapshot_xgb; $\tau=4$ is
byte-identical to Table 6's DAHS.

| $\tau$ | $J$ | 95% CI of mean | SFR | Median label entropy | Test rows kept |
|---:|---:|---|---:|---:|---:|
| 1 | 388.13 | [235.8, 566.3] | 0.0671 | 0.693 | 915 / 1600 |
| 2 | **372.31** | [225.1, 545.4] | **0.0653** | 0.649 | 1273 / 1600 |
| 3 | 373.38 | [225.5, 547.5] | 0.0657 | 0.638 | 1444 / 1600 |
| 4 | 381.42 | [232.7, 555.1] | 0.0689 | 0.638 | 1525 / 1600 |

Every cost interval overlaps every other. $\tau=2$ is the best point estimate;
deployed $\tau=4$ is third. On SFR, $\tau=1$ versus $\tau=4$ is a null
($p_{\mathrm{adj}}=0.34$). Longer rollouts do not buy measurable accuracy on the
default simulator. We do not read Table 9 as confirming an interior
$\tau^\star$ of the kind Proposition 1's worst-case bound would suggest.
Figure 4.

![Figure 4. Composite cost versus rollout horizon $\tau$. Source
`results/E4/tau_summary.parquet`.](../figures/E4/tau_composite_cost.png)

The companion sweep is $M \in \{1,5,10,20,40\}$. All five cells completed.
Figure 5 plots composite cost against $M$.

**Table 10.** Number of continuations. $M=20$ is the deployed model. Entropy is
in-band at every $M$. $M=1$ has identically zero rollout SE, so the
fraction of labels with separation below one SE is zero by construction.
$M=1$ post-calibration ECE $0.067$ misses the $0.05$ bar.

| $M$ | $J$ | SFR | Median entropy | frac $<1$ SE | Test kept | ECE pre$\to$post |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 380.24 | 0.0685 | 0.641 | 0 | 1150 / 1600 | 0.077$\to$0.067 |
| 5 | 380.44 | 0.0680 | 0.644 | 0.567 | 1384 / 1600 | 0.123$\to$0.034 |
| 10 | 382.40 | 0.0691 | 0.649 | 0.455 | 1457 / 1600 | 0.155$\to$0.027 |
| 20 | 381.42 | 0.0689 | 0.638 | 0.334 | 1525 / 1600 | 0.170$\to$0.021 |
| 40 | 382.78 | 0.0686 | 0.652 | 0.215 | 1526 / 1600 | 0.189$\to$0.018 |

Paired against $M=20$, $M=1$ differs by $-1.18$ in $J$ (95% CI $[-7.83, 5.18]$,
Wilcoxon $p=0.91$); $M=5$, $M=10$ and $M=40$ likewise include zero ($M=40$:
$+1.36$, $[-3.05, 5.99]$, $p=0.35$). The five-cell span on mean $J$ is about
$0.7\%$. Multi-sample labels improve calibration ECE and keep more test rows
through the ambiguity filter. They do not improve deployed cost. Why $M > 1$
remains the argument of
Section 4.3 --- the submitted labels were hindsight-optimal on one path, not
estimates of expected cost --- not a claim that $M=20$ is an accuracy lever.

![Figure 5. Composite cost versus number of continuations $M$. Source
`results/E4/n_samples_summary.parquet`.](../figures/E4/n_samples_composite_cost.png)

## 6.5 Robustness across untuned configurations

![Figure 6. Robustness grid across 12 untuned configurations (4 arrival rates
$\times$ 3 SLA tightnesses). Heat map of composite cost. The outlined cell is
the Table 6 default and matches it to machine precision.](../figures/E8/robustness_grid_heatmap_composite_cost.png)

Figure 6 is that heat map.

Four methods are frozen across the grid: DAHS, greedy_mpc, snapshot_xgb, Always-EEDD.
Always-COVERT is not on the grid. Teachers replan with each cell's true arrival
rate and tightness; DAHS and snapshot_xgb are the default-trained rankers with no
retraining. Among those four, greedy_mpc wins 8 of 12 cells and EEDD wins the four
light-load default/loose cells. DAHS wins none. The `arr1.65_default` cell
reproduces Table 6 exactly on every non-timing column for the four methods. The
grid is a stress test of transfer, not a second evaluation of the default ranking,
and it says the one-step teacher --- not the distilled student --- is the most
robust of the four under load and tightness changes the ranker was not retrained
on.

## 6.6 Calibration and interpretability

Isotonic calibration on a 20% held-out shift split improves expected calibration
error from 0.1700 to 0.0213 (acceptance threshold 0.05) and Brier score from
0.1730 to 0.1240, and *degrades* soft cross-entropy from 0.828 to 2.358. EDD
never wins a label on the calibration split and is passed through uncalibrated.
The reliability diagrams are Figure 7.

![Figure 7. Reliability diagrams before and after isotonic
calibration.](../figures/E5/reliability_pre_post.png)

Global SHAP values [@lundberg2017shap] on the 38-dimensional ranker input
(26-feature observation plus 12 regime posterior coordinates) rank
`queue_length` (0.370), `queue_length_lag_1` (0.152), `mean_slack_minutes`
(0.120), `interval_index_in_shift` (0.053) and `queue_length_lag_2` (0.052).
The sixth coordinate is a regime one-hot (`regime_post_9`, 0.040).
Congestion and slack dominate; the product-clock features do not. That is
consistent with Section 3.5: perishability binds, but the customer clock is the
more frequent pivot, and EEDD already reads both clocks inside the action.
Figure 8.

![Figure 8. Global SHAP feature importance for the
ranker.](../figures/E5/shap_summary.png)

## 6.7 Real-data grounding

**Fitting, not validating**. The submitted version set the simulator's input
parameters by choice and then compared them to a public trace. Fitting is the
right operation wherever a trace exists. Appendix C reports the candidate
families, the AIC table, and the selected family. Inter-arrivals are lognormal
(Olist, $n=98{,}241$); the exponential alternative is worse by $\Delta$AIC
$=34{,}502$. Customer windows are the closest shape match (KS $D=0.039$).
Processing time is not fitted: the trace's purchase-to-approval delay is not pick
time. Perishable fraction 0.20 is a design choice (Olist food/drink share is
0.0099). The operating arrival rate 1.65 / min is an operating point, not a fit.
Figure 9.

![Figure 9. Simulator input distributions against the Olist order trace
(mean-normalised).](../figures/A/olist_validation.png)

A frozen-ranker replay under empirical Olist arrivals (burstiness CV $2.68$,
against Poisson CV $1.00$) preserves the ranking: greedy_mpc 1783, DAHS 1815,
snapshot_xgb 1839, Always-EEDD 3806. Absolute costs inflate; the teacher still
leads and EEDD still trails. Figure 10.

![Figure 10. Method KPIs under Poisson against empirical-Olist bursty arrivals
(frozen rankers).](../figures/A2/olist_arrivals_compare.png)

## 6.8 Ablations

**Table 11.** Retrain and inference ablations versus DAHS, 50 paired shifts.
Composite cost is the column that decides. No ablation rejects equality with
DAHS after BH control on cost, SFR, or tardiness (7-arm family, including
`single_sample_rollout`). `random_ambiguity_filter` is per-shift identical to
DAHS by construction --- the deployed filter is never applied at evaluation ---
and is omitted from the table. Non-rejection is not an equivalence test.
The training wall-clock to convergence is reported for
retrain arms; inference-only arms do not retrain.

| Ablation | $J$ | $J$ vs DAHS [95% CI] | $p_{\mathrm{adj}}$ | SFR | Train wall (s) |
|---|---:|---|---:|---:|---:|
| hard_labels | 382.43 | $+$1.01 [$-$5.52, 7.13] | 0.83 | 0.0685 | 1302 |
| no_calibration | 381.46 | $+$0.04 [$-$5.76, 4.89] | 0.83 | 0.0685 | --- |
| no_regime | 383.56 | $+$2.14 [0.04, 4.61] | 0.34 | 0.0691 | 1124 |
| no_switching_controller | 380.43 | $-$0.99 [$-$4.24, 1.96] | 0.83 | 0.0687 | --- |
| `single_sample_rollout` ($M{=}1$) | 380.24 | $-$1.18 [$-$7.83, 5.18] | 1.00 | 0.0685 | --- |
| `top5_features` | 383.14 | $+$1.72 [$-$3.44, 6.59] | 0.83 | 0.0691 | 897 |
| DAHS | 381.42 | --- | --- | 0.0689 | --- |

The hard-label comparison is a null on both composite cost and service-failure
rate. Soft labels remain the default because they are the deployed configuration,
not because they outperform one-hot arg-max. Removing the switching controller
slightly *improves* point-estimate cost; the wrapper is a deployability guardrail
paid in KPI, not a performance component. The `top5_features` arm, trained on the
SHAP top five only, is likewise a null: the remaining 33 ranker coordinates are not earning
their dimensionality on this test set.

A single-sample rollout ablation ($M=1$ relabel and retrain) is the $M=1$ cell of
Table 10 and the `single_sample_rollout` row of Table 11 (byte-identical
parquets). It matches $M=20$ on $J$. The $\tau=1$ snapshot is Table 9.

## 6.9 On the PPO baseline

Untuned PPO at 8{,}000 epochs (`ppo_baseline`, `n_steps`$=64$) collapses to a
constant EEDD policy: its 50-shift cost is $695.770797$, identical to Always-EEDD
to machine precision. That is not the Table 6 `ppo_fair` row ($J=610.93$), which
is a separately trained stock Stable-Baselines3 policy with a full KPI record.
Sensitivity analysis: a 12-configuration grid was run on the same 50 test
shifts, so the winning configuration is not a train-split selection.

**Table 12.** PPO sensitivity grid. `gap_closed_fraction` $=$
$(\text{baseline}-\text{best})/(\text{baseline}-\text{DAHS}) = 0.783$. Factor
spreads: normalisation $313.5$; `n_steps` $22.5$; $\gamma$, GAE $\lambda$ and
entropy coefficient $0$.

| Configuration | $J$ |
|---|---:|
| 8 of 12 (including the untuned default) | 695.77 |
| `n_steps`$=256$ | 673.29 |
| obs-norm only | 540.66 |
| **obs-norm and reward-norm** | **449.60** |
| reward-norm only | 763.06 (Always-EDD) |

Tuning closes a substantial share of the gap. The reading that PPO's deficit is
structural rather than budgetary is withdrawn. The tuned configuration
(`norm(obs=True, rew=True)`, $J=449.60$) is the PPO baseline going forward; it
still trails DAHS ($381.42$). Table 6 reports the untuned `ppo_fair` row because
that is the policy evaluated with a full KPI record; quoting 450 without that
record would mix a sweep cost with a table of decompositions.

## 6.10 On the offline reinforcement-learning baseline

**Action coverage, and a correction.** The submitted behaviour policy was
round-robin over the pool. The interval index is an observed feature, so
$a = t \bmod |\mathcal{H}|$ is a deterministic function of the observation and
conditional coverage is one action per state by construction. The revision logs
under `random`. On 8,000 training states the effective number of actions is
5.999 of 6 overall and 5.995 of 6 on the breach-prone quartile; the
interval-conditional mean is 5.937. Coverage is adequate: the offline-RL deficit
cannot be attributed to unsupported actions in the hard region.

After an observe-once logger retrain (the previous logger called `observe()`
twice per interval and zeroed `n_arrivals_last_interval` on 31 of 32 states),
DAHS beats fitted Q on composite cost, $381.42$ versus $396.80$ (ratio $1.04$,
paired interval $[2.90, 29.83]$, confirmatory interval excludes 0). The margin is
small. It is real. Figure 11.

![Figure 11. Sample efficiency: DAHS versus the offline reinforcement-learning
baseline (fitted Q-iteration) at matched shift budgets.](../figures/E9/data_efficiency_offline_fqi.png)

Frozen across the four scenarios and the twelve-cell grid, FQI tracks DAHS
rather than collapsing. Under high-load-perishable both sit near 11,450--11,470
while PPO reaches 23,223 and Always-EEDD 24,062. Under default, FQI is the fifth
method in Table 6, between snapshot_xgb and Always-COVERT.

## 6.11 Model misspecification: labelling in one world, deploying in another

Proposition 2 bounds the accumulation of per-step model error as
$O(\varepsilon \tau^2)$ and predicts that the operationally best horizon
shortens as $\varepsilon$ grows. The experiment labels once under the nominal
simulator and evaluates frozen controllers --- DAHS, rolling_mpc, offline_fqi,
Always-EEDD, Always-COVERT --- under perturbed dynamics: arrival rate
$\times\{0.8,0.9,1.0,1.1,1.25\}$, processing time on the same grid, SLA
$\times\{0.8,1.0,1.25\}$, shelf life $\times\{0.8,1.0,1.25\}$, and picker count
$\Delta \in \{-2,-1,0,1,2\}$. The online teacher is pinned to the *nominal*
model, so it replans with a wrong model rather than with the truth.

**Table 13.** Mean relative-degradation slope of composite cost against the
perturbation, averaged over axes.

| Method | Mean slope | Rank (slower degradation first) |
|---|---:|---|
| COVERT | 18.7 | most robust (model-free) |
| offline_fqi | 22.0 | |
| DAHS | 22.8 | |
| rolling_mpc | 23.7 | |
| EEDD | 27.7 | least robust |

DAHS is not the most robust model-based method; FQI's mean slope is slightly
shallower. Always-COVERT, which carries no model, degrades slowest. Always-EEDD
degrades fastest on load, capacity and processing time. On the SLA axis DAHS has
the *steepest* slope of the five. Shelf-life slopes are near zero.

This experiment does not retrain at $\tau \in \{1,2,3,4\}$ as $\varepsilon$
grows, so it does not test the prediction that $\tau^\star$ shortens. Section 4.4
already noted that $\tau^\star$ enters the swept grid only once
$\varepsilon \gtrsim 0.29$; mild cells cannot show the optimum moving inward.
We report the slopes and that the $\tau$--$\varepsilon$ interaction is unmeasured.

## 6.12 Computational cost, and scaling in the size of the rule pool

### Offline cost

Walking each shift forward once and branching at each epoch (Section 4.3) costs

$$ N + N \cdot |\mathcal{H}| \cdot M \cdot \tau \quad \text{interval-steps per shift,} $$

linear in $N$ rather than quadratic in the submitted labeller, which replayed
each shift from $t=0$ for every candidate. For the deployed setup --- $N=32$,
$|\mathcal{H}|=6$, $\tau=4$, $M=20$, 250 training shifts --- labelling consumed
$4{,}401{,}600$ interval-steps ($3{,}668{,}000$ train, $733{,}600$ test) in
$758$ s train and $164$ s test wall-clock on the campaign machine (Intel Core
i9-14900K, 24 cores / 32 threads, 64 GB, Windows 11, Python 3.12.10). Rule
calibration, which sweeps $k$ under $M=5$, consumed $2{,}966{,}400$ additional
interval-steps. A separate single-thread `compute_budget measure` pass on three
shifts recorded $389.3$ interval-steps per second ($2.75$ h single-core for the
$3{,}848{,}000$-step corpus formula used by that driver). Labelling wall-clock
above is the labeller's own multi-core measurement and remains the campaign
figure.

### Online cost, and per-decision inference latency

**Table 14.** Mean per-decision inference latency, 50 default shifts.
The p95 column is the mean, across shifts, of each shift's per-decision p95
(so DAHS can have mean 3.66 ms and p95 3.60 ms). It is not a pooled p95 over
all decisions.

| Method | Mean (ms) | p95 (ms) | Wall-clock s / shift |
|---|---:|---:|---:|
| DAHS | 3.66 | 3.60 | 0.141 |
| snapshot_xgb | 3.30 | 3.13 | 0.129 |
| offline_fqi | 3.19 | 3.38 | 0.124 |
| ppo_fair | 0.17 | 0.25 | 0.014 |
| greedy_mpc | 588 | 866 | 18.8 |
| rolling_mpc | 645 | 900 | 20.7 |
| COVERT / EEDD / FIFO | $<10^{-3}$ | $<10^{-3}$ | 0.01 |

DAHS is $176$ times faster per decision than rolling_mpc (3.66 ms vs 645 ms) and
$161$ times faster than greedy_mpc. Static rules remain three orders of magnitude
faster still. The amortisation claim is this ratio, on named hardware, not an
argument that a forward pass is faster than a tree of simulations. At a
15-minute review interval, 645 ms is $0.07\%$ of the epoch, so the latency
product is real and operationally unused unless a tighter review is required.

Labelling the $M=20$, $\tau=4$ corpus took $922$ s ($15.4$ min) of wall-clock.
Ranker training (Stage 3) took $1.25$ h on the same machine. At
$20.7-0.14=20.6$ extra seconds per shift of online lookahead, labelling alone
is repaid after about $45$ shifts; labelling plus training is repaid after
about $250$ shifts. Both figures are hardware-specific and ignore that the
teacher is also the better policy.

### Scaling in $|\mathcal{H}|$

Labelling cost is linear in $|\mathcal{H}|$ under uniform allocation of $M$
continuations to every rule. Two mechanisms in the labeller address that.

Adaptive sample allocation (successive halving) discards clearly dominated rules
part-way through the $M$ continuations and spends the remaining budget on the
survivors. Hierarchical selection first screens a cheap one-step score and only
then rolls the shortlist to depth $\tau$. Successive halving is implemented
(`costs_at_epoch_successive_halving`). A 50-shift diagnostic at the deployed
$(M,\tau)$ produced arg-max agreement $0.856$ against uniform allocation, mean
label KL $0.185$, and a $1.1\%$ step saving (`successive_halving.json`); the
verdict is unsuccessful, so production labels stay uniform. Hierarchical
selection is described here and is not implemented. We did not
retrain DAHS at pool sizes $2$, $4$ and $8$.
The statement we can support is the complexity: uniform labelling is
$\Theta(N|\mathcal{H}|M\tau)$ interval-steps per shift; successive halving
replaces $|\mathcal{H}|M$ with $|\mathcal{H}|\log M$ in the usual approximation
when gaps are large relative to rollout SE. At $M=20$, $33\%$ of training epochs
have a best-versus-second gap below one pooled SE (Section 4.3), which is
exactly the regime in which early discarding is least safe.

---

# 7. Discussion

The campaign supports a smaller claim than the one we started with, and that is
the claim we are willing to defend.

On the default operating point, a rollout-trained selector beats every static
dispatching rule we screened, with Always-COVERT as the static to beat
($J=454$ against DAHS $381$, $49/50$ shifts) rather than Always-EEDD ($J=696$,
but only $21$ strict wins and $22$ exact ties). It beats fitted Q-iteration
($397$) and an 8{,}000-epoch PPO policy ($611$ in Table 6; $450$ once observations and
rewards are normalised; $696$ for the `n_steps`$=64$ collapse). It does not beat the teachers that generate its labels
($356$ and $363$). Distillation therefore loses $5$--$7\%$ of composite cost
against online truncated lookahead and returns a $176$-fold reduction in
per-decision latency.

The expensive parts of the method do not earn that $5$--$7\%$. $\tau=1$ matches
$\tau=4$ on service-failure rate and is not demonstrably worse on cost; $M=1$
matches $M=20$ on cost; every retrain ablation is a null; a random ambiguity
filter is identical to the deployed filter. WSPT, screened out of the default
pool, beats DAHS under high-load-perishable ($J=11{,}169$ against $11{,}427$,
interval excludes 0). What moves the default table is the
*existence* of a counterfactual per-rule cost vector, not the Monte Carlo
refinement of that vector past a single continuation or a single interval. A
practitioner who already has a resettable simulator and who can afford 645 ms
per decision should run the one-step teacher. A practitioner who needs a
millisecond decision can distil it, and should not expect the student to catch
the teacher.

Two comparisons that looked structural are not. PPO's deficit was a missing
normalisation wrapper (`gap_closed_fraction`$=0.783$). FQI's coverage under
round-robin was degenerate by construction; under a random behaviour policy
coverage is adequate and a 4% cost gap remains. Both baselines belong in the
table. Neither is a demonstration that value-based or policy-gradient methods
cannot do this task.

Perishability binds at this horizon (Table 1): 35.5% of decision epochs carry
an expiry-pivotal order in queue (7.3% of queued perishables; 1.4% of all
orders), against 95.1% of epochs carrying a due-pivotal order. FEFO --- the
true expiry-only rule --- drives spoilage essentially to zero at catastrophic
cost (Table 6). EEDD is the rule that reads both clocks. That does not
imply that a selector over those rules will win every regime: WSPT wins
high-load-perishable, and Always-EEDD is statistically tied with DAHS under
balanced load.

---

# 8. Limitations

### 8.1 Shared-simulator circularity

Every method in Table 6, teachers included, is labelled or trained in the same
simulator it is evaluated in. The comparison of training signals is internally
valid. It is not a claim about transfer to a physical warehouse. Section 6.11
perturbs the evaluation dynamics while keeping the labels nominal; that is a
start, not a substitute for a plant. A practitioner without a trustworthy
simulator should prefer value learning from logs (Section 2.4).

### 8.2 A small heuristic pool

Nine candidates were screened and six retained. Adaptive sample allocation
(successive halving) is implemented in the labeller; hierarchical selection is
described in Section 6.12 and is not implemented. We did not measure accuracy as
a function of pool size. The labelling cost grows with $|\mathcal{H}|$. A larger, generated pool --- genetic programming of the low-level
rules [@branke2016automated; @nguyen2017gpsurvey] followed by the same selector
--- is a different paper. EEDD's 65% win rate already warns that a small
hand-designed pool can concentrate.

### 8.3 A single warehouse setting

One layout, ten pickers, one-order tours, exogenous routing, Poisson default
arrivals. Appendix C fits inter-arrival shape to Olist; processing time and
shelf life are design parameters. Nothing here is a claim about multi-block
warehouses, batching, or picker routing. Section 2.1 stated that restriction
before any number.

### 8.4 No online adaptation after deployment

The ranker is frozen. Regime posteriors are computed from the fitted mixture,
not updated from live outcomes. A shift that drifts away from the training
mixture gets the wrong specialist. The misspecification grid (Section 6.11) is
the evidence we have that this matters, and Always-COVERT's shallower slope is
a reminder that a dumb static rule does not carry a wrong model.

Further limitations, stated so they are not rediscovered as objections.

*Partial observability.* The aliasing rate among $\phi$-identical training
states that disagree on the cost-minimising rule is $0.145$ on 1,927 pairs, with
mean regret $0.30$ and an estimated $3.7\%$ share of the achievable benefit
available from rule selection. That number is an upper bound on the part of the
residual regret attributable to $\phi$, not a reason to treat $\phi$ as a state.

*Regime order.* $K^\star=12$ is the edge of the swept grid. BIC is still
falling ($-168{,}901$ at $K=8$, $-222{,}525$ at $K=10$, $-240{,}139$ at
$K=12$). Mean ARI is $0.970$, so the clustering is reproducible; the model
order is not identified. We did not widen the grid, because doing so would
retrain Stage 3 onward.

*A pool of six is what a dispatcher already has.* Nine candidates were
considered and six retained (Section 3.6). Adding generated rules would change
the labelling budget and the action-coverage comparison, and we did not do it.

*The FQI comparison is a 4% cost gap after a logger correction.* It is not a
demonstration that bootstrapped value learning cannot close that gap with a
different model class or a different behaviour policy.

---

# 9. Conclusion and Future Work

We studied a selection hyper-heuristic for periodic-review warehouse
dispatching under two deadline clocks, trained by offline truncated rollouts of
a screened rule pool. The training mechanism is not new. The question was what
the supervision is worth.

On the default corpus, rollout labels at matched shift budgets beat fitted Q
and PPO, and beat every static rule, with Always-COVERT as the static champion
on composite cost. They do not beat online truncated lookahead. Single-sample labels match the
deployed $M=20$ configuration on composite cost. A one-step ($\tau=1$) label is
mixed on that cost and a null on service-failure rate. The operational product
is a 3.7 ms decision that spends
$5$--$7\%$ more than a 645 ms teacher.

The most useful extension is the one the dynamic-dispatching literature already
runs: replace the hard truncation of Propositions 1 and 2 with a learned value
tail [@ulmer2019offlineonline; @goodson2017rolloutframework], and let the
horizon be state-dependent. That construction would likely permit a shorter
$\tau$ --- attractive here, because Table 9 already says extra depth is not
buying accuracy and Proposition 2 says extra depth accumulates model error.
Two other extensions follow from the campaign rather than from taste: widen the
regime grid until $K^\star$ is interior, or drop the mixture; and test the
$\tau$--$\varepsilon$ interaction that Section 6.11 left unmeasured.

---

# Appendix A. State features, their provenance, and their redundancy

## A.1 Where each feature comes from

The submitted version listed the feature names and nothing else --- not where they
came from, and not whether they had been screened. Neither question had an answer,
because the set was designed rather than selected and no redundancy analysis was
performed. This appendix supplies both. The table is generated directly from
`simulation.state_extractor.FEATURE_PROVENANCE` by
`experiments/feature_analysis.py`, so the manuscript and the deployed feature map
cannot drift apart.

The observation is $\phi(S_t) \in \mathbb{R}^{26}$. It is an observation, not a
state (Section 3.2).

| # | Feature | Group | Source / rationale |
|---:|---|---|---|
| 1 | `queue_length` | queue | Standard congestion state; de Koster et al. (2007). |
| 2 | `mean_queue_age` | queue | Waiting-time proxy; distinguishes fresh from stale backlog. |
| 3 | `max_queue_age` | queue | Tail of the waiting-time distribution --- starvation detector. |
| 4 | `pct_critical` | queue | Share of queue within 30 min of its due time. |
| 5 | `pct_perishable` | queue | Gates the expiry-rule mask; tells the selector when the product clock is live at all. |
| 6 | `n_arrivals_last_interval` | queue | Short-run demand shock; wave arrivals, Boysen et al. (2019). |
| 7 | `labor_utilization` | resources | Classical queueing load indicator rho. |
| 8 | `n_pickers_busy` | resources | Absolute capacity remaining this epoch. |
| 9 | `mean_pickup_time_recent` | resources | Realised service-rate estimate; drifts with order mix. |
| 10 | `n_orders_late_so_far` | deadline | Realised failures to date; regime indicator. |
| 11 | `n_orders_at_risk_30min` | deadline | Count with negative slack inside 30 min --- the actionable set. |
| 12 | `mean_slack_minutes` | deadline | Mean d - t - p; the ATC/MS/COVERT decision variable. |
| 13 | `std_slack_minutes` | deadline | Slack dispersion; separates uniform from bimodal pressure. |
| 14 | `mean_processing_time_remaining` | deadline | Expected work content of the queue. |
| 15 | `pct_high_priority` | deadline | Share of the economically weighted tail. |
| 16 | `pct_expiring_30min` | expiry | Product-clock analogue of pct_critical (revision). |
| 17 | `mean_expiry_slack` | expiry | Mean x - t - p over perishables; FEFO's decision variable (revision). |
| 18 | `n_spoiled_so_far` | expiry | Realised spoilage to date (revision). |
| 19 | `arrival_rate_recent_60min` | arrivals | Non-stationarity detector; empirical lambda-hat. |
| 20 | `queue_length_lag_1` | history | Congestion trend; standard lag structure. |
| 21 | `queue_length_lag_2` | history | Congestion trend. |
| 22 | `queue_length_lag_3` | history | Congestion trend. |
| 23 | `failure_rate_lag_1` | history | Failure trend; replaces breach_rate lag under the new metric. |
| 24 | `failure_rate_lag_2` | history | Failure trend. |
| 25 | `failure_rate_lag_3` | history | Failure trend. |
| 26 | `interval_index_in_shift` | temporal | Position in the finite horizon; end-of-shift effects. |

## A.2 What changed in this revision, and why

**Two features are removed.** `time_to_next_expected_carrier` was computed as
$1/\lambda$ and was therefore *constant* within any one configuration --- zero
variance, zero information. `intervals_remaining` was an exact affine function of
`interval_index_in_shift`, the two summing to $N = 32$ by construction. Together
they made the feature matrix exactly singular. That is not a cosmetic defect: the
regime layer (Section 4.5) fits a full-covariance Gaussian mixture on these
columns, so the covariance was singular, only the ridge term `reg_covar`
prevented a failure, and each additional mixture component bought likelihood by
collapsing further onto the degenerate directions. The submitted BIC curve fell
monotonically to the edge of its grid and "selected" $K = 6$ at the boundary,
which is an artefact of the degeneracy rather than evidence of six regimes. A
correlation analysis would have caught both features before they reached the
model; none was run.

**Three features are added**, all on the product clock: `pct_expiring_30min`,
`mean_expiry_slack` and `n_spoiled_so_far`. The product deadline now enters the
objective (Section 3.3), and a selector cannot act on a constraint it cannot
observe --- in particular FEFO's own decision variable, expiry slack, was not
visible to the ranker that had to decide when to deploy FEFO.

**One feature is renamed.** The breach-rate lags become `failure_rate_lag_1..3`,
tracking the change of primary metric in Section 3.3.

## A.3 Redundancy analysis

`experiments/feature_analysis.py` runs four diagnostics on the training feature
matrix: near-constant columns; Pearson and Spearman correlation with pairs above
$|r| = 0.95$ flagged; variance inflation factors, obtained by regressing each
feature on the remaining ones; and correlation-distance clustering with one
nominated representative per cluster.

No column is near-constant. Ten pairs exceed $|r|=0.95$, of which
`labor_utilization` with `n_pickers_busy` is $r=1$ (they encode the same
capacity residual) and the failure-rate lags with `n_orders_late_so_far` are
$r \ge 0.979$. Fourteen features have VIF $\ge 10$: `queue_length`,
`n_arrivals_last_interval`, `n_pickers_busy`, `labor_utilization`, the three
queue-length lags, the three failure-rate lags, `n_orders_late_so_far`,
`mean_queue_age`, `max_queue_age`, and `mean_slack_minutes`. The clustering
nominates dropping `labor_utilization`, `mean_queue_age`, `queue_length_lag_1`,
`queue_length_lag_2`, and the three failure-rate lags. We did not drop them
from the deployed model: the `top5_features` ablation (Section 6.8) already
asks whether the long tail of the map earns its keep, and it does not, at
least not detectably on 50 test shifts. The VIF column is a warning that
coefficient-level stories about individual features are not identified, which
is why Section 6.6 reports SHAP on the fitted trees rather than linear weights.

After dropping the two degenerate features of Section A.2, the regime BIC still
selects $K^\star=12$ at the edge of
$\{2,3,4,5,6,7,8,10,12\}$ (mean ARI $0.970$, above the $0.85$ threshold). BIC
values fall from $+36{,}871$ at $K=2$ to $-240{,}139$ at $K=12$ and do not turn.
The grid chose $K$, not a mode in the data. Input dimension is $38$.

# Appendix B. Configuration and hyperparameters

Every value below is read from `config.yaml` in the accompanying repository, which
is the source the campaign shares, except where a grid winner is named.

**Fitted Q-iteration.** `config.yaml` default trees are `max_depth`$=4$,
`n_estimators`$=200$, `learning_rate`$=0.05$, $\gamma=0.99$, 20 iterations.
The E9 grid winner written to `results/E9/hp_winner.json` is `max_depth`$=4$,
`n_estimators`$=500$, `learning_rate`$=0.05$, $\gamma=0.9$. Table 6 uses the
winner.

**Simulator.** 8-hour shift; $N = 32$ review intervals of $L = 15$ minutes;
$m = 10$ pickers; queue capacity 200 orders, with overflow recorded as rejected
demand rather than discarded. Arrivals are Poisson at a nominal 1.65 orders/minute. Order
attributes: processing time Triangular$(2, 5, 12)$ min; customer window
$d_o - a_o$ Triangular$(15, 45, 90)$ min; shelf life $x_o - a_o$
Triangular$(20, 60, 120)$ min for the 20% of orders that are perishable, with
$x_o = \infty$ otherwise; priority classes $\{$low, medium, high$\}$ drawn at
$(0.50, 0.35, 0.15)$ with economic weights $w_o \in \{1, 2, 4\}$. Section 3.4
gives the provenance of each.

**Objective** (Section 3.3). $W_b = 3.0$ per late shipment, $W_t = 0.2$ per minute
of lateness, $W_s = 5.0$ per spoiled order, $W_h = 0.005$ per order still queued at
shift end; every per-order charge multiplied by $w_o$. The submitted configuration
had no $W_s$ term, applied no $w_o$, and priced an abandoned order at $0.005$
against $3.0$ for one served late.

**Shift corpora.** Seeds drawn from one `SeedSequence` (root 42) and partitioned
into three disjoint contiguous blocks: 250 training, 30 calibration, 50 test. The
calibration block is new in this revision and exists so that ATC's and COVERT's
look-ahead scales can be fitted without touching the training or test shifts.

**Labelling** (Section 4.3). Rollout horizon $\tau = 4$; $M = 20$ independent
continuations per state-rule cell under common random numbers, with the per-cell
standard error recorded alongside every label; behaviour policy `random` --- **not**
round robin, because the interval index is itself an observed feature, which made
the submitted round robin a deterministic function of the state (Section 6.10).
Tempered softmax with a **per-row** temperature $T(s) = \beta\,\sigma(s)$, where
$\sigma(s)$ is the standard deviation of that state's own cost vector and $\beta$
is searched so that the median training-label entropy
falls within $[0.216, 0.505] \times \log|\mathcal{H}|$. Campaign fit:
$\beta=0.470$, median entropy $0.638$ nats, band $[0.387, 0.905]$. Expiry-rule
mask threshold 0.05 on the perishable fraction (a no-op on the screened pool,
which contains no expiry-only rule). Test-corpus ambiguity filter at
$\theta = 2.2/|\mathcal{H}|$, never applied to the training corpus. The
hard-label ablation of Section 6.8 replaces the tempered softmax with
the one-hot arg-max of the same cost vector and is otherwise identical.

**Rule pool** (Section 3.6). Nine candidates before screening: FIFO, EDD, EEDD,
FEFO, WSPT, ATC, MS, MDD, COVERT. Stage 1 retained
`[EEDD, COVERT, MS, ATC, MDD, EDD]` and fitted ATC $k^\star=3.0$ (portfolio;
standalone $1.5$) and COVERT $k^\star=4.0$.

**Regime layer** (Section 4.5). Gaussian mixture, full covariance, with $K$
selected by BIC over $K \in \{2,3,4,5,6,7,8,10,12\}$ and 5 EM restarts per $K$;
stability checked by the mean adjusted Rand index over 10 refits against a 0.85
threshold. $K^\star=12$ at the grid edge, mean ARI $0.970$.

**Ranker.** Gradient-boosted trees [@chen2016xgboost], `multi:softprob` objective,
sample-weighted by inverse label entropy. Hyperparameters selected from an
18-configuration grid (`max_depth` $\in \{4,6,8\}$ $\times$ `n_estimators` $\in
\{200,500,1000\}$ $\times$ `learning_rate` $\in \{0.03,0.1\}$) by 5-fold
cross-validation grouped on `shift_id`; isotonic calibration on a 20% held-out
shift split. Selected: `max_depth`$=6$, `n_estimators`$=200$,
`learning_rate`$=0.03$. Test ECE after isotonic: $0.0213$ (pre $0.1700$).

**Switching controller** (Section 4.7). Minimum dwell $T_{\min} = 2$ intervals;
entropy gate at half the maximum entropy.

# Appendix C. Fitting the input distributions to a real order trace

**Source.** Olist Brazilian e-commerce public dataset [@olist2018dataset],
approximately 100k orders. Inter-arrival times are computed as within-day
differences of order timestamps, which removes the multi-day growth trend that
otherwise dominates the series. Because the trace is measured in days and the
simulator in minutes, all comparisons are on mean-normalised samples: the trace
fixes distribution *shape*, and the operating rate is set separately.

Candidate families for inter-arrival times: exponential, lognormal, gamma,
Weibull. Selected: **lognormal** ($n=98{,}241$, AIC $161{,}984$). Exponential is
worse by $\Delta$AIC $=34{,}502$. Shape tests against the simulator's Poisson
default: KS $D=0.153$, Wasserstein-1 $0.443$; simulated CV $1.00$ versus real
$2.68$. Poisson is therefore an operating point, not a fitted arrival process.
Section 6.7's Olist-bootstrap replay is the robustness check that uses the
empirical burstiness.

Customer windows: triangular \((15, 45, 90)\) after rescaling the
purchase-to-estimated-delivery distribution. Closest shape match of the three
inputs (KS $D=0.039$, subsampled $p=0.022$, $W_1=0.035$).

Processing time: not fitted. The trace field is purchase-to-approval, not pick
time. The triangular \((2, 5, 12)\) is a literature three-point standard
[@tompkins2010facilities]. Shape test against that proxy is correspondingly
poor (KS $D=0.686$).

Perishable fraction: design parameter $0.20$ against Olist food/drink share
$0.0099$. Shelf life has no public analogue.

---

\section*{Data availability}

The simulator, training pipeline, configuration and result artifacts that
produced every number in this paper are in the accompanying repository. CAOR
requires that the data and code underlying the work be deposited (Guide for
Authors, Option C). A persistent archive with a DOI will be linked at
acceptance; until then the versioned source is
\url{https://github.com/Vittal-Mukunda/Disruption-Aware-Heuristic-Scheduling}.

\section*{CRediT authorship contribution statement}

**Vittal Mukunda:** Conceptualization, Methodology, Software, Formal analysis,
Writing --- original draft. **Atharva Somani:** Software, Investigation,
Validation, Data curation. **Pranjal Malaiya:** Methodology, Writing --- review
and editing, Supervision.

\section*{Declaration of competing interest}

The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

\section*{Funding}

This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

\section*{Acknowledgements}

We thank the editors and the anonymous referees of the original submission for
comments that changed the model, not only the prose.
