# Response to Reviewers — CAOR-D-26-01812

**Training Signals for Warehouse Dispatching-Rule Selection: A Controlled
Comparison**

Vittal Mukunda, Atharva Somani, Pranjal Malaiya

---

## Summary of the revision

We thank the four reviewers for reviews that were detailed, technically precise,
and — on several points — correct about defects we had not seen. The revision is
substantial, and we want to be direct about its character before going through
the items individually.

**Four of the comments each independently invalidated every number in the
submitted paper.** Reviewer 2 showed that the objective discounted an abandoned
order by a factor of 600 against one served late, and that the reported breach
rate excluded abandoned orders from its denominator entirely (R2.1). Reviewer 2
also showed that the rollout labels were single-path realisations rather than
estimates of expected cost (R2.3). Reviewer 1 showed that ATC was never
calibrated, and that the pool was neither motivated nor screened (R1.4c, R1.4d).
Reviewers 1 and 2 together showed that perishability entered no constraint and no
cost, and that the rule we called FEFO was in fact EDD (R1.1c/d/f, R2.2).

Each of these changes the data-generating process or the objective. We therefore
**rebuilt the model and regenerated every quantitative result**. No number from
the submitted version is carried forward into a claim; where a submitted table is
reproduced, it is explicitly marked superseded and retained only as the evidence
for a diagnosis.

**We also withdraw the paper's novelty claim.** Reviewer 1 (2.b, 2.c, 2.e) and
Reviewer 5 are right that simulating a rule pool offline and fitting a classifier
to the result is not new: it is rollout classification policy iteration in the
reinforcement-learning literature and multi-pass rule selection in the scheduling
literature. Section 2 now places the method inside both traditions, corrects our
mischaracterisation of the dispatching-rule-selection literature, and withdraws
the claims that rollouts are "normally used online" and that our method "inverts
the usual deployment". The contribution is reframed as an empirical study — a
comparison of training signals on one shift corpus — not a new
training mechanism.

**Two defects we found in our own audit, which no reviewer named, caused symptoms
several reviewers flagged.** The objective ignored the priority weights that WSPT
and ATC rank by, so those rules were graded against a criterion they were not
optimising; and the dispatcher admitted orders arriving up to fifteen minutes in
the future, reserving and idling a picker for them, which structurally penalised
every arrival-agnostic rule and never penalised FIFO. Together these explain the
counterintuitive results Reviewer 1 asked about in comment 6.a. Both are fixed.

**Finally, the headline margin is not the submitted 3.8× and is not the 1.2×
that the metric correction alone produced on the old logs.** Charging unserved
orders on the submitted event logs compressed the FIFO gap from about 3.8× to
about 1.2×. After regenerating under causal admission, the live default-scenario
FIFO gap is **3.89×** on composite cost (manuscript Table 5: DAHS $J=382.27$,
FIFO $J=1486.82$). Teachers remain cheaper than DAHS ($356$--$363$). We state
the live ranking in Section 6.2 before leaning on any method-to-method story.
The relative comparison between training signals survives; the size of the
static-rule margin does not, and the submitted paper overstated it.

A short summary of what changed structurally:

| Change | Prompted by | Where |
|---|---|---|
| Objective: unserved-and-overdue orders charged; spoilage priced; priority weights applied | R2.1, R1.1c, R1.6a | §3.3 |
| Two independent deadline clocks; true FEFO; EDD and EEDD under their own names | R1.1f, R2.2 | §3.1, §3.6 |
| Causal periodic-review admission (no picker reserved for future arrivals) | own audit; explains R1.6a | §3.4 |
| Labels are Monte Carlo means over $M$ continuations, with per-cell standard errors | R2.3 | §4.3 |
| Rule pool calibrated, expanded to nine candidates, screened to six | R1.4a–e | §3.6, §6.1 |
| Sequential decision model in Powell's framework; POMDP stated explicitly | R5.1, R5.3, R2.4 | §3.2 |
| Related work repositioned; novelty claim withdrawn | R1.2a–e, R5.2 | §2 |
| Rolling-horizon MPC teacher added as a baseline | R2.6 | §5, §6.2 |
| PPO sensitivity sweep; offline-RL action coverage corrected | R1.6b | §6.9, §6.10 |
| Model-misspecification experiment and Proposition 2 | R2.5 | §4.4, §6.11 |
| Composite cost as the primary metric throughout | R1.6c | §3.3, §6 |
| Terminology and notation section | R5.4 | §1.1 |

---

# Reviewer 1

We are grateful for a review that engaged with the method at the level of its
mechanics. Several of these comments identified defects in the code, not only in
the prose, and we say so where that is the case.

## 1. Problem setting

### 1.a — Positioning within order-picking research

> *"The problem can be positioned within order-picking research provided that
> routes, batching, and travel-time estimation are clearly identified (and
> justified) as exogenous or deliberately excluded."*

**Agreed, and added.** New **Section 2.1** states the scope explicitly: this paper
addresses order release and dispatching only. Storage assignment and layout are
fixed; batching is fixed at one order per pick tour; routing and travel-time
estimation are exogenous, entering the model only through the distribution of
processing times $p_o$, in the three-point form standard when only time-standard
data are available.

We give three justifications — the planning horizons of the decisions involved
differ by orders of magnitude; rule selection is only well-posed once the
downstream problems are fixed, or the comparison confounds the rule with the
batching policy it is paired with; and embedding a travel-time estimator would
make the rule comparison a comparison of travel models. We also state the cost of
the restriction (DAHS cannot exploit batching or routing synergies) and record it
in Section 8.3.

### 1.b — The warehouse literature review was one sentence

> *"There is no exploration of the problem features and of the methodologies
> used, namely dispatching rules and data-centric methods."*

**Agreed.** New **Section 2.2** covers the classical dispatching-rule families and
their properties (arrival-driven, due-date-driven, processing-time-driven, and
composite slack/processing indices), the warehousing-specific literature on order
release under stochastic arrivals, and the recent data-centric direction. New
**Section 2.3** covers simulation-trained rule selection specifically, which is
the tradition this method belongs to.

### 1.c — Perishability and priority class did not appear in the objective

> *"...each order has a due time, a perishability flag and a priority class, but
> the latter two do not appear in the objective function (Eq. 1), nor in any
> constraints."*

**Correct, and this was a defect rather than an omission in the write-up.** Both
now enter the objective (**Section 3.3**):

- **Priority class.** WSPT and ATC have always ranked by $w_o/p_o$, but the
  submitted objective weighted every order equally. Those two rules were being
  graded against a criterion they were not designed for, which is a substantial
  part of why their reported performance looked anomalous — see our response to
  6.a. Every per-order charge is now multiplied by $w_o$.
- **Perishability.** A spoilage term $W_s$ is added. Without it,
  "perishability-constrained" was not a property of the optimisation problem at
  all.

### 1.d — Does perishability bind at a 15-minute horizon?

> *"In what percentage of decisions does delaying an order by one interval (15
> minutes) alter its feasibility, quality, or economic value?"*

**We adopted your criterion directly and pre-registered a threshold before
running it.** New **Section 3.5** defines an order as *expiry-pivotal* at epoch
$t$ when $t + p_o \le x_o < t + L + p_o$ — one interval of delay is the difference
between saleable goods and waste. Three conditions were fixed in advance, all of
which had to hold for the framing to stand. Measured over 7,440 recorded
decision epochs on the 30-shift calibration block $\times$ 8 behaviour rules
(empty queues omitted; repeated measures, not 7,440 independent shifts):

| Quantity | Measured | Pre-registered threshold |
|---|---:|---:|
| Decisions with an expiry-pivotal order in queue | **35.5%** | ≥ 5% |
| Perishables whose expiry binds before their due date | **27.6%** | ≥ 10% |
| Epochs where the rule choice changes the spoilage count | **91.0%** | ≥ 10% |

All three are met, so the framing stands. We report two qualifications with it
rather than after it: only **1.4%** of individual orders and **1.3%** of economic
weight are expiry-pivotal at any epoch, so perishability is not the dominant cost
driver — the customer clock is, with 95.1% of epochs carrying a due-pivotal order.
What makes the product clock decision-relevant is concentration and frequency: the
pivotal orders cluster, and at 91.0% of epochs the rule choice moves realised
spoilage, by about four orders where it moves it at all.

Had the thresholds not been met, we would have dropped the perishability framing
and said so. We wrote the criterion down before we ran it for exactly that reason.

### 1.e — Order-level FEFO requires justification

> *"FEFO normally applies to inventory lots, not to customer orders. Has
> inventory already been allocated? How is one expiry value calculated for a
> multi-item order?"*

**Both questions now answered explicitly in Section 3.1**, under *"Where the
expiry of an order comes from"*. We model the stage **after lot allocation**: an
upstream allocation policy — exogenous in the same sense as routing and batching —
has already committed specific lots to specific orders, so each order inherits a
concrete expiry from the stock reserved against it. For a single-line order that
is the allocated lot's expiry. For a multi-line order, the order ships as a unit
and is therefore constrained by its most perishable component, so
$x_o = \min_{\ell \in o} x_\ell$ over the allocated lines. The experiments use
single-line orders, so $x_o$ is the lot expiry directly; the minimum-over-lines
rule is the generalisation and requires no change to the controller, which reads
only $x_o$.

### 1.f — FEFO is not deadline-aware

> *"The authors call FEFO 'deadline-aware', but FEFO has to do with perishability
> (expire date), not with the order's due time (due date)."*

**You are exactly right, and this was a naming error in the code as well as the
prose.** The submitted `fefo()` sorted on `sla_due`. That is EDD. Both rules now
exist under correct names: FEFO sorts on the product deadline $x_o$, EDD on the
customer deadline $d_o$. Results are reported accordingly, and Section 3.6 states
plainly that the rule which produced the submitted FEFO results is EDD.

Separating them exposed something the submitted model could not have shown, and
we think it is the most useful thing to come out of this comment. With two
deadlines, *neither* single-clock rule is the right one. EDD ignores expiry
entirely. FEFO ranks on $x_o$, which is $\infty$ for the 80% of orders that are
not perishable, so it sorts every non-perishable order behind every perishable one
— on this order mix, close to a strawman, and the reason the FEFO mask had to
exist at all. We therefore added **EEDD**, which sorts on the *effective deadline*
$\min(d_o, x_o)$ — whichever clock binds first. It is the rule a scheduler would
write once told that orders carry two deadlines, and Section 6.1 shows it is by
some distance the strongest rule in the pool.

## 2. Contribution and positioning

### 2.a — The stated intersection is narrow and inconsistent with the contributions

### 2.b — "Offline rollout distillation" is not new

> *"It appears closely related to established rollout classification policy
> iteration... (see Lagoudakis and Parr (2003) and further research on it)."*

### 2.c — Simulation-trained dispatching-rule selectors already exist

> *"Creating the training set with the so-called 'multi-pass' is used since Wu
> and Wysk (1988) until Mouelhi-Chibani and Pierreval (2010), Shiue et al.
> (2020), and many others."*

### 2.d — The characterisation of Đurašević and Jakobović is inaccurate

> *"I do not know of any selector trained by genetic programming... Prior
> selectors have been trained essentially by supervised learning, just like the
> authors do in the current paper."*

### 2.e — The novelty claim should be substantially revised

> *"The manuscript should substantially revise its novelty claim, engage with
> these literatures, and clarify whether the contribution is primarily the
> warehouse application, the specific system integration, or an empirically
> demonstrated advantage rather than a new training paradigm."*

**We accept all five points and have rewritten Section 2 accordingly.** Taking
them together, since our response is one position:

**The mechanism is not novel and we no longer claim it is.** Section 2.4 states
that what we do — estimate action values by simulation at a sample of states, then
fit a classifier to represent the improved policy — **is** Rollout Classification
Policy Iteration, citing Lagoudakis & Parr (2003) and the subsequent work of Fern,
Yoon & Givan, Dimitrakakis & Lagoudakis, and Farahmand et al. Section 2.3 traces
the same construction in the scheduling literature from Wu & Wysk (1988) through
Mouelhi-Chibani & Pierreval (2010) to Shiue et al. (2020).

**We withdraw the two specific claims you identify as misleading.** The statements
that rollouts are "normally used online" and that DAHS "inverts the usual
deployment" are removed from Section 2 *and* from the Introduction, where the
first of them also appeared.

**We correct our mischaracterisation of the selector literature.** Section 2.3
now states that genetic programming in this literature is used predominantly to
*generate* the low-level rules that are subsequently selected among, not to learn
the selector, and that prior selectors — including those in Đurašević &
Jakobović, Mouelhi-Chibani & Pierreval, and Shiue et al. — are trained essentially
by supervised learning on simulation-derived labels, which is what we do as well.
We state that the method belongs squarely inside that tradition rather than
departing from it.

**On what the contribution now is** (your 2.e asks us to choose): Section 2.6
frames it as an **empirical study**, with three components — a controlled
comparison of training signals at matched data budgets, holding environment,
corpus, model class, feature set and objective fixed and varying only how the
supervision is built; a warehouse instantiation with two deadline clocks where the
second one's relevance is *measured* rather than assumed; and a sample-efficiency
result. We state explicitly: *"We make no claim to a new training paradigm, and
the contribution should be read as the application and the controlled comparison,
not the mechanism."*

Two details of our instantiation differ from standard RCPI and we note them as
details, not contributions: the classifier is fitted to the full per-action cost
vector rather than the arg-max, and the rollout is truncated with the truncation
error bounded explicitly.

**References added:** Lagoudakis & Parr (2003); Fern, Yoon & Givan;
Dimitrakakis & Lagoudakis; Farahmand et al.; Wu & Wysk (1988); Mouelhi-Chibani &
Pierreval (2010); Shiue et al. (2020); Branke et al. and Nguyen et al. on GP for
rule generation; Klapp et al.; Ulmer (several); Goodson et al.; Powell.

## 3. The DAHS method

### 3.a — How were the state features identified and selected?

> *"How were they identified (based on the literature?) and selected (was there
> any correlation analysis?)"*

**The honest answer for the submitted version is that they were designed, not
selected, and no correlation analysis was performed.** We say so, and the omission
turned out not to be cosmetic.

**Appendix A** is rewritten in three parts. A.1 is a per-feature provenance table
— every feature with its group and the literature source or design rationale it
derives from — generated directly from the code so the manuscript and the deployed
feature map cannot drift apart. A.3 reports the redundancy analysis that was
missing: near-constant columns, Pearson and Spearman correlation with pairs above
$|r| = 0.95$ flagged, variance inflation factors, and correlation-distance
clustering.

**Running that analysis found two degenerate features.**
`time_to_next_expected_carrier` was computed as $1/\lambda$ and was therefore
constant within any configuration — zero variance, zero information.
`intervals_remaining` was an exact affine function of `interval_index_in_shift`,
the two summing to $N = 32$ by construction. Together they made the feature matrix
exactly singular, which silently corrupted the regime layer: a full-covariance
Gaussian mixture on singular data buys likelihood by collapsing onto the
degenerate directions, which is why the submitted BIC curve fell monotonically to
the edge of its grid and "selected" $K = 6$ at the boundary. Both features are
removed, the $K$ sweep now runs on a grid wide enough to turn ($K \in
\{2,\dots,12\}$ against the submitted $\{3,\dots,6\}$) with five EM restarts per
$K$, and selection at a grid endpoint is reported as such.

Three expiry-pressure features are added, since the product deadline now enters
the objective and a selector cannot act on a constraint it cannot observe. The
observation is $\phi(S_t) \in \mathbb{R}^{26}$.

After `feature_analysis` (`results/S1_features/feature_analysis.json`): no
near-constant columns; ten pairs with $|r|\ge 0.95$, including
`labor_utilization` vs `n_pickers_busy` at $r=1$; fourteen features with
$\mathrm{VIF}\ge 10$ (queue length and its lags, picker occupancy and
utilisation, failure-rate lags, late-so-far, queue age, slack). Coefficient-level
stories about single features are not identified; Section 6.6 reports SHAP on
the trees instead. After dropping the two degenerate features, BIC still
selects $K^\star=12$ at the edge of $\{2,3,4,5,6,7,8,10,12\}$ (mean pairwise
ARI $0.970$, above $0.85$). BIC falls from $+36{,}871$ at $K=2$ to $-240{,}139$
at $K=12$ and does not turn (`runs/phase4/phase4_regime.json`). Ranker input
dimension is $38$ (26 features plus 12 regime posteriors). We did not widen
the $K$ grid.

### 3.b — Where do the 1600 test states come from?

**Answered in one line in Section 4.3**, under *"Where the corpora come from"*.
Shift seeds are drawn from a single `SeedSequence` and partitioned into three
disjoint blocks — training, calibration, test. Each shift contributes one decision
state per review interval, so a block of $n$ shifts yields $32n$ states: the test
block of 50 shifts gives $50 \times 32 = 1600$ before filtering. In the submitted
version, 865 of those survived the ambiguity filter; under the corrected labels
1525 of 1600 survive.

The calibration block is new in this revision and exists so that rule
hyperparameters can be fitted without touching training or test shifts — part of
why ATC went uncalibrated before.

## 4. The pool of dispatching rules

This comment did more to change the paper than any other, and we treat its five
parts as one programme of work. New **Section 6.1** reports the results; they run
on the 30-shift calibration block, which is disjoint from both training and test.

### 4.a / 4.d — Motivate the pool; it could be expanded, with screening

**The pool is now constructed to span the information sources a dispatcher can
key on** (Section 3.6), expanded from four rules to **nine candidates** — FIFO,
EDD, EEDD, MS, MDD, FEFO, WSPT, ATC, COVERT — and screened, with the screening
table reported as you allow.

### 4.b — Why would FIFO add value in a due-date oriented setting?

**It does not, and it is dropped.** On the calibration corpus FIFO is the
cost-minimising rule at **0.1%** of decisions and its marginal contribution is
identically **zero**. It entered as the zero-information control and the screen
reports that it earns nothing.

We note separately that FIFO's flattering fourth place in the submitted Table 1 is
explained by the admission defect described under 6.a: the dispatcher penalised
every arrival-agnostic rule and never penalised FIFO.

### 4.c — ATC was not calibrated; WSPT cannot beat a calibrated ATC

> *"WSPT is a special case of ATC (when k tends to infinity), which implies that
> ATC should not be outperformed by WSPT... the authors should calibrate it to
> both cases, explain how they did it, and report the final parameter value."*

**You are right, and the code confirmed it: `atc_lookahead_k` was fixed at 2.0 and
no search over $k$ existed anywhere in the repository.** We now calibrate on the
disjoint calibration block, twice, exactly as you ask — once for standalone use,
because ATC is itself a reported benchmark, and once for portfolio contribution,
which is what matters when the rule sits inside a selector. COVERT is calibrated
the same way.

| Rule | $k^\star_\text{standalone}$ | cost at $k^\star$ | $k^\star_\text{portfolio}$ | deployed |
|---|---:|---:|---:|---:|
| ATC | 1.5 | 459.2 | 3.0 | **3.0** |
| COVERT | 4.0 | 404.3 | 4.0 | **4.0** |

**This settles the inversion you identified.** ATC's standalone cost is U-shaped
in $k$ with a minimum of 459.2 at $k^\star = 1.5$, rising monotonically to
**1004.2 at $k = 20$** — a factor of 2.19. Since WSPT is exactly the
$k \to \infty$ limit, that curve *is* the ATC-to-WSPT interpolation, and it shows a
fitted ATC beating WSPT by more than two-fold. The submitted finding that WSPT won
32% of decisions against ATC's 10% was an artefact of the unfitted scale, not a
property of the rules — precisely your diagnosis.

The two optima also differ by a factor of two, which is the concrete case for
calibrating both: the scale that makes ATC best *alone* is not the scale that makes
it most useful *inside a pool*, where its job is to cover states the others handle
badly.

### 4.e — Complementarity must be shown across the state space, not across shifts

> *"Instead, they should present the rules' performance across key dimensions of
> the state space (queue state and deadline pressure)."*

**Agreed, and this is the comment whose answer is least favourable to us.** The
submitted Figure 1 varied the *instance*, not the state, and could not establish
what we claimed from it. Figure 1 is replaced by win rate over a grid of exactly
the two dimensions you name — queue length and deadline pressure (mean slack), in
quantile bins. We also replaced win-rate screening with **marginal contribution**:
the increase in achievable cost when a rule is removed, with a bootstrap interval,
because win rate alone cannot distinguish a redundant rule from a specialist.

**Screening results** (30 calibration shifts; retained when the marginal-contribution
interval excludes zero):

| Rule | win rate | marginal contribution | 95% CI | retained |
|---|---:|---:|---|:--:|
| EEDD | 0.650 | 5.403 | [4.840, 5.991] | ✓ |
| COVERT | 0.145 | 2.047 | [1.613, 2.527] | ✓ |
| MS | 0.070 | 0.248 | [0.119, 0.412] | ✓ |
| ATC | 0.055 | 0.086 | [0.032, 0.155] | ✓ |
| MDD | 0.011 | 0.039 | [0.014, 0.070] | ✓ |
| EDD | 0.068 | 0.007 | [0.000, 0.021] | ✓ |
| FIFO | 0.001 | 0.000 | [0.000, 0.000] | — |
| WSPT | 0.000 | 0.000 | [0.000, 0.000] | — |
| FEFO | 0.000 | 0.000 | [0.000, 0.000] | — |

**And the state-space grid does not support a strong complementarity claim.** Over
the 4×4 grid, **EEDD owns 15 of the 16 cells** and COVERT the sixteenth. The best
single rule (always EEDD) wins 65.00% of decisions; the per-cell oracle wins
72.29%. The gap a state-conditioned selector has to work in, at that resolution, is
**7.29 percentage points**. We report this in Section 6.1, before the main
comparison rather than after it, with two qualifications that bound it in the other
direction: the grid oracle is a *floor* on available headroom rather than a ceiling
on the method, since DAHS reads 26 features and not two binned ones; and win rate
is not the objective, which is why the screen uses marginal contribution.

We also report that EEDD's 0.650 win rate exceeds our own pre-registered 0.60
concentration ceiling. We kept the rule and reported the number rather than
dropping it to pass our own gate.

We did **not** compute a composite-cost oracle on the same 4×4 grid. The
win-rate oracle gap over always-EEDD is 7.29 percentage points (per-cell oracle
72.29% vs always-EEDD 65.00%; EEDD owns 15 of 16 cells). On the test-set
objective the static champion is Always-COVERT, not Always-EEDD: DAHS $J=382.27$
against COVERT $455.21$ (49/50 shifts). That is the gap DAHS captures against
static rules. It does **not** capture the teacher gap: greedy_mpc $356.98$,
rolling_mpc $363.42$. Distillation amortises a slightly worse scoring rule.

## 5. Data instance generation

### 5.a / 5.b — Parameters were set arbitrarily, then validated against real data

> *"Why not fit the input distributions to the real data?"*

**Agreed — that is the right order of operations and we had it backwards.**
Section 3.4 now gives a provenance table in which every input is either fitted to
data, grounded in a cited source, or declared a design choice; none is left
unexplained. Appendix C reports the fitting procedure and the candidate families
compared by AIC.

We are also explicit about what the public trace **cannot** be fitted to. The
Olist dataset is e-commerce order metadata with no warehouse pick-time field, so
processing time is grounded in the published three-point time-standard convention
for manual picker-to-parts picking rather than fitted; the submitted comparison
against purchase-to-confirmation latency ($D = 0.685$) was not a valid test of
anything, and we say so. No public trace carries a product expiry, so shelf life
is a declared design parameter and is swept rather than defended.

Fits are in `results/A/validation_stats.json` and Appendix C. Inter-arrivals:
Olist is bursty (CV $2.68$) against the simulator's Poisson (CV $1.00$); a
lognormal on the trace beats an exponential by $\Delta$AIC $=34{,}502$; KS
$D=0.153$. Poisson at 1.65 / min remains an **operating point**, not a fitted
arrival process. Customer windows: triangular $(15,45,90)$ after rescaling is
the closest shape match of the three inputs (KS $D=0.039$, subsampled $p=0.022$,
$W_1=0.035$) and corroborates the design triple. Processing time is **not**
fitted — the trace field is purchase-to-approval, not pick time; KS $D=0.686$
against triangular $(2,5,12)$ is a proxy test and is poor. Perishable fraction
$0.20$ is a design parameter against Olist food/drink share $0.0099$. A
frozen-ranker replay under empirical Olist arrivals preserves the ranking
(greedy still leads, EEDD still trails; Section 6.7).

### 5.c — The triangular parameters were not disclosed

**Now stated in the main text**, in the Section 3.4 provenance table rather than
only in an appendix: processing time Triangular$(2, 5, 12)$ min, customer window
Triangular$(15, 45, 90)$ min, shelf life Triangular$(20, 60, 120)$ min.

## 6. Results

### 6.a — Counterintuitive results lack explanation

> *"How does WSPT perform worst in throughput? It should particularly succeed in
> that metric... FIFO has the fourth best composite cost."*

**Both are artefacts of the environment and the objective, not properties of the
rules, and finding the causes was the most useful consequence of this comment.**
Section 6.2 gives them under *"Two anomalies in the submitted results, and their
causes"*.

**Cause 1 — the objective did not measure what the rules optimise.** WSPT and ATC
rank by $w_o/p_o$ using the priority weights of Section 3.1; the submitted
objective weighted every order equally. They were correctly maximising weighted
throughput while the scoreboard counted unweighted breaches. Fixed by putting
$w_o$ into the objective (your comment 1.c).

**Cause 2 — the dispatcher idled pickers on behalf of arrival-agnostic rules.**
The submitted simulator admitted every order arriving before the *end* of the
current interval — fifteen minutes of look-ahead — and set start times to
$\max(\text{picker free}, a_o, t)$. Ranking a not-yet-arrived order therefore
**reserved a picker and left it standing idle** until that order appeared. FIFO,
sorted by arrival, never triggered this; WSPT and ATC, which are arrival-agnostic,
triggered it constantly. This is what a picker utilisation of 0.686 alongside a
queue of roughly 180 waiting orders was recording — a picker cannot be idle a
third of a shift with that much work available unless the dispatch model idles it.
It also explains FIFO's flattering composite cost, since FIFO was the only rule the
mechanism never penalised.

Section 3.4 replaces this with a properly causal periodic-review admission rule —
only orders arrived by $t$ are eligible at $t$ — which also removes fifteen minutes
of undisclosed look-ahead from the observed state.

Under the corrected admission rule, on 50 default test shifts, WSPT records the
**highest** throughput ($743.3$ vs FIFO $730.2$) and every method sits at picker
utilisation $\approx 0.956$ (WSPT $0.955$). Cause 2 is confirmed. Composite-cost
ranking is manuscript Table 5: greedy_mpc $356.98$, rolling_mpc $363.42$, DAHS
$382.27$, Always-COVERT $455.21$, Always-EEDD $696.62$, FIFO $1486.82$
($3.89\times$). Mean arrived $=791$ (identical across methods on each seed;
a terminal admit at $T$ counts arrivals in $(T-L,T]$ as unserved; they are
never dispatched); dropped $=0$.

### 6.b — The RL failure explanations are post hoc

> *"...the authors should report sensitivity to the discount factor, GAE
> parameter, rollout horizon, entropy coefficient, and reward normalization; and
> quantify offline action coverage in breach-prone states."*

**You are right that the submitted evidence did not support the conclusion drawn
from it.** We varied exactly one thing — the training budget — and concluded the
deficit was "structural, not budgetary". That did not follow. Worse, the
implementation used **no observation normalisation and no reward normalisation**
on a feature vector whose columns span queue lengths in the hundreds and
utilisations in $[0,1]$, which is among the most common causes of a
policy-gradient method failing to learn at all.

**We now run exactly the sweep you name** (Section 6.9): discount $\gamma$, GAE
$\lambda$, rollout length `n_steps`, entropy coefficient, and the full $2\times2$
over observation and reward normalisation — swept jointly rather than marginally,
because observation scaling changes the effective gradient magnitude while reward
scaling changes the advantage scale. Every configuration is trained for 8{,}000
timesteps (Stable-Baselines3 `total_timesteps`, not review epochs)
and evaluated on the same held-out shifts. The grid was scored on those test
shifts first. A later calibration re-score of the same twelve cells picked the
same winner; nine of twelve calibration costs are identical, so that split
barely discriminates. The summary statistic is the fraction of the PPO-to-DAHS
gap the best configuration recovers.

**On offline action coverage, we found an error in our own argument and it is
instructive.** The submitted paper stated that because the behaviour policy was a
uniform round robin, "every action is covered at every state". That is false. The
behaviour policy was $a_t = \mathcal{H}[t \bmod |\mathcal{H}|]$ — and the interval
index $t$ is itself one of the observed state features. The logged action was
therefore a **deterministic function of an observed feature**: marginally each rule
was taken equally often, but *conditional on the state* exactly one action was ever
observed, and $Q(s,a)$ for every other action was pure extrapolation. Marginal
uniformity concealed conditional degeneracy.

We therefore regenerate the corpus under a seeded random behaviour policy, and
report coverage — overall and **restricted to breach-prone states**, defined from
the labels as those whose best achievable rollout cost lies in the top quartile,
which is exactly where a value function most needs data. Both corpora are reported
so the effect of the behaviour policy on the baseline can be read directly.

**Sections 6.9 and 6.10 report the measurements, not a pre-registered new
baseline.** Tuning closed a material part of the PPO gap; coverage under
`random` is adequate. The structural reading is withdrawn. The tuned PPO cell
does **not** replace Table 5. The grid was scored on the test shifts first;
the calibration re-score later picked the same cell.

**PPO: the structural reading is withdrawn; Table 5 still reports the untuned
row.** Against the Always-EEDD collapse row ($J=696.62$) the sweep recovered
$78.3\%$ of the collapse-to-DAHS cost gap (`gap_closed_fraction`$=0.783$).
Against Table 5's named `ppo_fair` row ($J=611.77$) the same best cell closes
$0.70$. We do not call both denominators "untuned." Best cell: observation
*and* reward
normalisation (cost $450.45$ vs collapse $696.62$ vs DAHS $382.27$).
The grid was scored on the **test** shifts first; a later calibration re-score
picked the same cell. The tuned cell is a sensitivity result, not a
pre-registered baseline; manuscript Table 5 keeps
`ppo_fair` ($J=611.77$). A separate `ppo_baseline` run at `n_steps`$=64$
collapses to Always-EEDD ($J=696.62$). Coverage under `random`
(`results/E11_rl_sensitivity/action_coverage.json`): $5.999$ effective actions
overall, $5.995$ in breach-prone states, $5.937$ conditional on interval index.
Adequate. Fitted Q was retrained after the Stage-4 logger double-called
`observe()` and zeroed arrivals; a 4% cost gap remains ($397.64$ vs $382.27$).

### 6.c — The composite cost should be the primary metric

> *"I assume that all the algorithms were optimized for composite cost, but even
> that was unclear."*

**Agreed on both counts.** The composite cost is now the primary metric throughout
Section 6, tables are ranked by it, and Section 5 states explicitly that it is the
criterion every learned method optimises. The service-failure rate is reported as
its headline component. We also added an objective-weight sensitivity analysis
(Section 6.11 / `e4_sensitivity weights`) so that the ranking's dependence on the
weights is measured rather than assumed.

## 7. Minor issues

| | Comment | Resolution |
|---|---|---|
| **7.a** | DAHS acronym undefined | Defined on first use and in the Section 1.1 terminology table: *Disruption-Aware Heuristic Scheduling*. |
| **7.b** | P3 paragraph duplicates the first paragraph of the page | Deleted. |
| **7.c** | Paragraph titles in §2, §5, §6.7 end with ".." | Fixed; a check for it runs in our verification script. |
| **7.d** | "A reviewer will reasonably ask…" (P18, P22, P29) | All removed. The paper no longer addresses reviewers anywhere in its prose; the substance of those passages is retained, the second-person framing is not. |
| **7.e** | Reference 2 is incomplete | Completed: Dokeroglu, Kucukyilmaz & Talbi (2024), *Hyper-heuristics: A survey and taxonomy*, Computers & Industrial Engineering 187, 109815. |

---

# Reviewer 2

This review identified the two defects that most affected the paper's
conclusions. We are grateful for the precision of both.

## 1. The penalty weights create a loophole

> *"With $W_b = 3.0$, $W_t = 0.2$, and $W_u = 0.005$, a single completed late
> order is penalized as heavily as 600 unfinished orders... the controller can
> artificially lower the reported SLA-breach rate simply by ignoring difficult
> orders... I need to see revised results where every overdue order (whether
> completed late or left unfinished at the end of the shift) is counted as a
> breach."*

**Your arithmetic is right, the loophole is real, and this is the most important
correction in the revision.** Your supporting observation was right too: DAHS
completed 721.6 orders against FIFO's 750.6.

**Three changes** (Section 3.3):

1. **Unserved orders are charged.** For an order still waiting at the horizon we
   set $f_o = T + p_o$ — the earliest it could possibly finish, since it still
   requires a full pick. Every arrived order then has a well-defined outcome
   whether or not it was served. An unserved order past its deadline is charged
   exactly as a late one, at $W_b$. The $+p_o$ makes dispatching an order onto a
   free picker cost precisely what abandoning it costs, with any earlier dispatch
   costing strictly less, so doing the work is weakly optimal by construction —
   which the submitted objective did not guarantee. $W_h = 0.005$ survives strictly
   as a work-in-progress holding cost.

2. **The breach-rate formulae are stated explicitly**, as you ask, with their
   denominators named. We report a **service-failure rate** over *arrived* orders
   as the primary component, and both breach rates — over arrived and over
   completed orders — under names that make the denominator unambiguous, so the two
   versions of the paper remain comparable. Orders rejected at the door when the
   queue was at capacity are included in $\mathcal{A}$: they are real demand that
   went unmet, and excluding them would reopen the same gap elsewhere.

3. **The full outcome partition** — arrived, served, unserved, rejected — is
   reported as columns in manuscript Table 5, the live comparison. The
   side-by-side against the submitted scoreboard is gone from this revision:
   the article is now a standalone study, not a diff. Mean arrived $=791$,
   dropped $=0$ (identical across methods on each seed). A terminal admit at
   $T$ counts arrivals in $(T-L,T]$ as unserved; they are never dispatched.

**What this costs us.** Recomputing your metric on the submitted repository's own
per-order event logs (ten shifts, frozen model):

| | breach rate over completed orders | failures over *arrived* orders |
|---|---:|---:|
| DAHS | 3.10% | 15.00% |
| FIFO | 11.75% | 17.97% |
| PPO (submitted logs; then billed as matched budget) | 9.40% | 16.44% |

On those old logs the advantage over FIFO narrows from roughly **3.8× to 1.20×**,
and over PPO from 3.0× to 1.10×. That diagnosis is about the submitted
denominator, not the regenerated campaign.

**Live Table 5** (50 default test shifts, corrected objective and causal
admission): DAHS $J=382.27$, SFR $0.0669$, spoil $0.0384$, tardiness $0.754$,
throughput $731.9$, utilisation $0.956$. FIFO $J=1486.82$ ($3.89\times$), SFR
$0.1837$. Always-COVERT $455.21$ is the static to beat. Always-EEDD $696.62$
(DAHS strictly cheaper on 21/50, EEDD on 7, 22 exact ties). Teachers remain
cheaper (greedy $356.98$, rolling $363.42$). Snapshot $\tau=1$ is $388.97$
with a confirmatory paired interval that includes 0. Under the same paired
rule $\tau=2$ is cheaper than deployed $\tau=4$ ($J=373.16$; $-9.11$,
$[-17.61,-1.35]$). On the 12-cell robustness grid five methods are frozen:
greedy_mpc wins 8 cells, Always-EEDD wins 4, DAHS wins 0. Distillation is
$158\times$ faster ($4.24$ ms vs $670$ ms); $670$ ms is $0.07\%$ of a
15-minute epoch.

## 2. The handling of perishable orders is vague

> *"Does a perishable order have a distinct expiration timestamp, or is its
> expiration identical to the SLA due date?... Is a spoiled order included in the
> breach count? How are unfinished perishable items penalized when the shift
> ends?"*

**All four questions are now answered explicitly in Section 3.3**, under
*"Spoilage mechanics, stated explicitly"*. Taking them in order:

*Distinct expiry, or the due date?* **Distinct.** $x_o$ is drawn independently of
$d_o$, so for a perishable order either clock can bind first. In the submitted
model there was no $x_o$ at all and "spoilage" was defined as a perishable order
missing $d_o$ — which made the two events the same event by construction, exactly
the ambiguity you identified.

*What happens when an order spoils?* Its goods become unsaleable at $x_o$. The
simulator is discrete-interval: spoilage is assessed when the potential or a
KPI is computed, not as a continuous-time event at $x_o$. Objective spoilage
is $f_o > x_o$ with $f_o = T + p_o$ if unserved. Dispatch after expiry does
not clear $W_s$. The KPI uses the clock at $T$ (no $+p_o$). The order is **not**
removed from the queue: spoiled stock still has to be pulled
and disposed of, so it continues to consume a picker.
Keeping it also closes an incentive gap — if spoiled orders vanished, a controller
could free capacity by stalling until perishables expired, which is the same class
of loophole as your comment 1.

*Is a spoiled order in the breach count?* Lateness and spoilage are **separate
predicates** on the same order; an order may be neither, either, or both. When
both fire the composite cost charges both, since they are distinct economic losses
— a late shipment and destroyed stock. In the reported metrics they are kept apart
so nothing is double-counted in a headline: the breach rate counts lateness only,
the spoilage rate spoilage only, and the service-failure rate counts an order once
if it is late *or* spoiled.

*Unfinished perishables at shift end?* Through the same $f_o = T + p_o$ convention.
If $T + p_o > x_o$ the order is spoiled and charged $W_s w_o$; if also
$T + p_o > d_o$ it is late and charged $W_b w_o$ plus tardiness. Since shelf life
is at most 120 minutes against a 480-minute shift, a perishable still waiting at
shift end is spoiled with near certainty — which is intended: abandoning perishable
stock is the most expensive thing this objective can do.

## 3. Single-path rollouts do not estimate expected cost

> *"...each state-rule pair is evaluated using only a single pre-sampled future
> path. This approach only identifies the best rule for that specific
> realization... the rollout procedure must average over multiple independent
> continuations."*

**You are right, and the situation was worse than "only one sample".** In the
submitted implementation every stochastic quantity was pre-sampled when a shift was
constructed, and the labeller replayed *that same shift* from its start for each
candidate rule. All rules therefore saw **the identical realised future**. The
label recorded which rule was best in hindsight on one path, not which had the
lowest expected cost, and the rollout variance was **identically zero**. It also
means the bias–variance argument in the submitted Section 4.4 had no variance term
anywhere in the implementation — the mechanism we invoked to explain an interior
optimal horizon did not exist in the code.

**What we changed** (Section 4.3):

- A `branch(rollout_seed)` operation deep-copies the state at $t$, discards the
  not-yet-arrived tail, and resamples it from a stream seeded on
  $(\text{shift}, t, m)$ — genuine Monte Carlo, still bit-reproducible.
- The label is $\hat{J}^\tau_h(s_t) = \frac{1}{M}\sum_m \hat{J}^\tau_{h,m}(s_t)$
  with $M = 20$, and **the per-cell standard error
  $\widehat{\mathrm{se}}_h(s_t)$ is recorded alongside every label**, as you ask.
- **Common random numbers**: the continuation seed depends on shift, epoch and
  sample index and deliberately *not* on the rule, so every candidate is scored
  against an identical set of futures. The comparison is paired, and since the
  tempered softmax reads only *differences*, the shared arrival shock cancels.
- Proposition 1 is restated with truncation bias and estimator variance separated.

This also got cheaper, not more expensive: walking each shift forward once and
branching at each epoch is $O(N)$ where the submitted replay-from-zero was
$O(N^2)$, and removing that term is what paid for the $M$ samples.

**A caution we will report either way.** On a smoke corpus, **76.8%** of decision
epochs had their best and second-best rules separated by less than one pooled
standard error at $M = 20$. If that holds at full scale, the soft labels are
largely noise, and we will report it as a finding about the method rather than
smooth it over. It is why Section 6.4 now includes a sweep over $M$.

On the full training corpus at deployed $M=20$ (`data/label_meta.json`): mean
rollout SE $3.71$, median SE $1.81$, $\mathrm{frac}<1\,\mathrm{se}=0.334$,
median entropy $0.638$ (in band). The $M$ sweep completed at
$\{1,5,10,20,40\}$. Deployed cost sits in a $0.7\%$ band: $M=1$ $J=381.08$,
$M=20$ $382.27$, $M=40$ $383.63$. Paired against $M=20$, every cell's interval
includes 0 ($M=40$: $+1.36$, $[-3.05,5.99]$, $p=0.35$). $\mathrm{frac}<1\,\mathrm{se}$
falls as $M$ grows ($0$ at $M=1$ by construction; $0.567$, $0.455$, $0.334$,
$0.215$ at $5,10,20,40$). Multi-sample labels improve calibration ECE and keep
more test rows; they do not improve deployed $J$. The smoke-corpus $76.8\%$
figure does **not** hold at full scale; one-third of $M=20$ epochs remain
unseparated at one SE, which is still the reason the labels are soft.

## 4. The 25-D feature vector is an observation, not a Markov state

> *"...two entirely different queue configurations could map to the exact same
> 25-D vector while exhibiting completely different transition dynamics... The
> paper needs to distinguish the true full state from its feature representation
> ... or reframe the problem as a POMDP."*

**We accept this entirely and have reframed the problem.** New **Section 3.2**
distinguishes the true state $S_t = (\mathcal{Q}_t, \mathbf{b}_t, t)$ — the queue
with every order's full attribute tuple, the picker-availability vector, and the
clock — from the observation $x_t = \phi(S_t) \in \mathbb{R}^{26}$. The submitted
paper called $x_t$ "the state"; that was wrong.

**We do not claim $\phi$ is a sufficient statistic. It is not, and we give a
constructive witness.** Two queues of two orders each, differing only in which
order carries the tight deadline, agree on every coordinate of $\phi$ — same
length, same ages, same mean and standard deviation of slack, same mean processing
time, same critical and at-risk counts — yet incur different cost under the same
rule. `experiments/observability_analysis.py` searches a grid, verifies the feature
vectors coincide to machine precision *before* comparing anything, and reports the
gap. The largest ATC gap on that searched grid is **3.79 against −0.01** at
processing times $(4, 18)$ with one picker. Simulator processing time is
Triangular$(2,5,12)$, so $p=18$ is off-support; on-support cells have smaller
ATC gaps. The pair shows that $\phi$ can match while ATC cost depends on the
latent pairing. It is not a reachable-state regret floor.

Two conditions turn out to be necessary and we state both as part of the
construction, because they sharpen what the defect actually is. The queue must
**contend** for a picker — a ranking expresses a preference only when something has
to wait, so with a picker free per order every rule produces the same trajectory.
And the rule must key on slack and processing time **jointly**: the two queues
carry the same *multiset* of slacks. EDD sorts by due date, not slack; EEDD, MS
and MDD are closer slack-family rules, and no witness is claimed for them. ATC
and COVERT separate the pair. The precise defect is therefore that **$\phi$
retains the marginal distributions of slack and of processing time and destroys
their coupling.**

**Consequences, stated in both directions.** The unfavourable one is an
**irreducible regret floor**: two states $\phi$ cannot separate must receive the
same action, so whenever their optimal actions differ some regret is incurred that
no data or capacity can remove. We measure this rather than assume it away, over
mutual near-neighbours in standardised $\phi$-space, and report it in Section 8.
The favourable one is more specific: the rollout **labels** are computed from the
true state $S_t$, so the supervision target is *correct* and only the covariates
are lossy. The learning problem is regression with insufficient covariates — a
Bayes risk induced by the feature map — not a biased target. A richer $\phi$ (a
permutation-invariant set encoding over queued orders) is the remedy and is
recorded as future work.

The policy class is described as what it is: a **policy-function approximation**
over a hand-crafted belief summary, with no belief state maintained.

The empirical aliasing diagnostic (`results/S1_observability/observability.json`)
on 8,000 training states: 1,927 neighbour pairs, aliasing rate $0.145$, mean
regret $0.30$, mean regret given disagreement $2.08$, estimated share of
achievable benefit $0.037$. A constructed witness exists for ATC (cost gap
$3.8$) and COVERT ($0.8$); not for EEDD, MS, MDD, or EDD. $\phi$ is not a
sufficient statistic. The $3.7\%$ share is an upper bound on the part of residual
regret attributable to the feature map, not a reason to treat $\phi$ as a state.

## 5. Model misspecification is ignored

> *"Any model misspecification will inevitably corrupt the rollout labels, and
> these errors will compound over the 4-interval rollout horizon... While
> Proposition 1 addresses truncation error, it completely ignores model error."*

**Agreed, and we have added both the theory and the experiment.**

**Sketch 2** (the submitted draft labelled this Proposition 2) is a heuristic
envelope for kernel error, not a theorem: a finite $\bar{C}$ is not proved
under Poisson arrivals, and the implemented window cost is a terminal
potential. If one treats $|c|\le\bar{C}$ under probability-TV, the factor $2$
belongs in $\Gamma$. The first-order condition of that envelope is
$\tau^\star = 1/(2\varepsilon) - 1/2$, never $\tau^\star \approx 1/\varepsilon$.
We do not invert the vacuous scale into an operator map. Truncation bias still
falls linearly in $\tau$ while model error grows quadratically --- that *shape*
is what Section 4.4 is for.

**Section 6.11** tests it exactly as you suggest. We label under nominal parameters
and evaluate the frozen controllers under perturbed dynamics along four axes an
operator would have to estimate and would estimate imperfectly: arrival rate,
processing-time scale, due-window scale, and picker headcount. Nothing is
retrained. Two design points make it informative: the online lookahead controller
is pinned to the **nominal** model, so it plans with exactly the same wrong dynamics
DAHS's labels were built from — giving it the perturbed dynamics would hand it a
model nobody has; and the static rules are included as the misspecification-free
reference, since they carry no model at all and their degradation measures the
environment getting harder rather than any method getting worse.

We also state the limit of the result plainly in Section 8.2: we cannot estimate
$\varepsilon$ for a real warehouse, so the bound tells an operator how to trade
$\tau$ against model quality but not what their model quality is.

Mean relative-degradation slopes of composite cost, averaged over axes
(`results/E10_misspecification/degradation_summary.json`): Always-COVERT $18.7$
(most robust; model-free), fitted Q $21.9$, DAHS $22.7$, rolling_mpc $23.6$,
Always-EEDD $27.7$ (least robust). On the SLA axis DAHS is the steepest of the
five. The experiment does **not** retrain at $\tau\in\{1,2,3,4\}$ as
$\varepsilon$ grows, so it does not test the sketch's prediction that
$\tau^\star$ shortens. Section 4.4 already notes that $\tau^\star$ enters the
swept grid only once $\varepsilon\gtrsim 2/9$; mild cells cannot show the
optimum moving inward. We report the slopes and that the $\tau$--$\varepsilon$
interaction is unmeasured.

## 6. The online rollout MPC baseline is missing

> *"DAHS is trained to mimic a 4-step rollout, but we don't see how it compares to
> simply running that 4-step rollout directly online... evaluate all four rules
> over four intervals at each decision epoch, implement the best rule for one
> interval, and then replan."*

**This was the sharpest omission in the submitted paper: the thing DAHS claims to
amortise was never evaluated.** The submitted `greedy_mpc` was $\tau = 1$, while
the deployed model distils $\tau = 4$, so we could not say whether DAHS approaches,
matches or exceeds its own teacher — which is the whole amortisation argument.

`baselines/rolling_horizon_mpc.py` implements exactly the procedure you describe:
at each epoch evaluate every rule over $\tau$ intervals averaged over $M$
continuations, commit the arg-min for one interval, replan. It is reported in
manuscript Table 5 with KPIs **and per-decision wall-clock**, and the break-even — the number
of decisions after which the one-off labelling cost is repaid by the per-decision
saving — is reported in Section 6.12.

**A note on the text of this comment.** In the copy of the review we received, this
item ends mid-sentence: *"Including this benchmark would answer several critical
questions:"* with the list truncated. We have answered the four questions we believe
were intended — whether DAHS matches its teacher, what the online controller costs
per decision, whether the amortisation is worth it, and how each degrades under
misspecification — and Section 6.2 marks them as our reading. **We would be glad to
address the specific questions if the complete comment can be supplied.**

Live KPIs, 50 default shifts: greedy_mpc ($\tau=1$, $M=5$) $J=356.98$;
rolling_mpc ($\tau=4$, $M=5$) $J=363.42$; DAHS $J=382.27$. Teachers beat the
student; both paired intervals exclude 0. Mean per-decision latency: DAHS
$4.24$ ms, rolling_mpc $670$ ms ($158\times$), greedy_mpc $594$ ms. Labelling
the $M=20$, $\tau=4$ corpus took $922$ s wall-clock; at $21.33$ extra seconds
per shift of online lookahead, labelling alone is repaid after about $43$
shifts and labelling plus Stage-3 training after about $250$. Both figures are
hardware-specific and ignore that the teacher is also the better policy. The
teachers use $M=5$ at evaluation; labels were built at $M=20$. A matched $M=20$
teacher evaluation is in `results/E_teacher_M20`: rolling_mpc moves from $363.42$
to $363.76$; greedy_mpc is unchanged at $356.98$.

---

# Reviewer 3

We are grateful for a review that focused on what a practitioner would need in
order to adopt this, and for the remark about transparently reported negative
results — we have tried to hold that standard through a revision in which more of
the results are unfavourable than before.

## 1. Computational cost of the offline rollouts

> *"What is the total number of simulated steps? How long does labeling take on
> standard hardware?"*

**Now measured rather than estimated, and reported in Section 6.12.** The submitted
labeller reconstructed each decision state by replaying the shift from $t = 0$
separately for every rule, costing
$|\mathcal{H}|\left(\frac{N(N-1)}{2} + N\tau\right)$ interval-steps per shift —
for the setup you quote, 2,496 per shift and **624,000 in total**. The dominant
term is the replay and it buys nothing: it re-derives a state the shift walk has
already passed through.

Walking each shift forward once and branching at each epoch costs
$N + N|\mathcal{H}|M\tau$ per shift — linear in $N$ rather than quadratic.
Removing the $O(N^2)$ term is what pays for the $M$ continuations that make the
label an estimator at all (Reviewer 2, comment 3).

`experiments/compute_budget.py` computes the comparison analytically and measures
throughput on named hardware; `scripts/campaign_budget.py` reports the whole
campaign's budget per stage.

Labelling $M=20$, $\tau=4$, $|\mathcal{H}|=6$, 250 train + 50 test shifts
consumed $4{,}401{,}600$ interval-steps ($3{,}668{,}000$ train, $733{,}600$
test) in $758$ s train and $164$ s test wall-clock on the campaign machine
(Intel Core i9-14900K, 24 cores / 32 threads, 64 GB, Windows 11, Python
3.12.10). Rule calibration under $M=5$ consumed $2{,}966{,}400$ additional
interval-steps. A separate single-thread `compute_budget measure` pass on three
shifts recorded $389.3$ interval-steps per second ($2.75$ h single-core for
that driver's $3{,}848{,}000$-step corpus formula). Fitted Q logs the same
shift walk once; its training budget is the FQI iteration grid in Appendix B
(winner: depth 4, 500 trees, $\gamma=0.9$), not a second copy of the rollout
tree. The two training budgets are therefore not a matched simulation-step
count, and we do not claim they are.

## 2. Scalability in the size of the heuristic pool

> *"The impact of pool size on offline training time and sample efficiency;
> whether sub-sampling or hierarchical selection could mitigate this cost."*

**Section 6.12 distinguishes two costs that grow differently**, which we think is
the useful form of the answer:

- **Labelling cost is linear in $|\mathcal{H}|$** — every rule is rolled out at
  every state. Going from four rules to twenty multiplies the offline budget by
  five, with *no* change to the online cost of the deployed selector: the ranker
  emits one more logit and nothing else moves.
- **Statistical cost is worse than linear.** An $|\mathcal{H}|$-rule pool is an
  $|\mathcal{H}|$-class problem, so the corpus must support that many decision
  boundaries. Sample efficiency degrades in $|\mathcal{H}|$ even though the
  per-state supervision stays dense, and the *shift* budget at which the selector
  saturates is the quantity to watch, not the step budget.

**We implement successive halving as a compute-budget diagnostic, not as the
production labelling path.** `label_one_shift` calls uniform `costs_at_epoch`.
Successive halving spends the same total budget adaptively --- a cheap round over all rules, discard the
worst fraction, reallocate to the survivors. This is compatible with the soft label
rather than in tension with it: the tempered softmax maps a clearly inferior rule
to near-zero probability however precisely its cost was estimated, so a rule
eliminated in round one loses nothing it would have contributed.

**Hierarchical selection is described but not implemented, and we say so.** The
rules partition naturally by information source (arrival-driven,
customer-deadline-driven, product-deadline-driven, processing-composite), so a
two-stage selector could choose a family and then a member. With six screened rules
the flat selector is not the bottleneck, so we record it as the natural next step
for a pool of twenty rather than claim it.

We did **not** retrain DAHS at pool sizes $2$, $4$ and $8$. Analytic labelling
cost is linear in $|\mathcal{H}|$ (`pool_scaling_analytic.parquet`: 4 rules
$2.57$M steps, 8 rules $5.13$M, 20 rules $12.8$M at the deployed
$(N,M,\tau)$). Successive halving at the deployed $(M,\tau)$ on 50 shifts,
using a nine-candidate screen set (not the deployed $|\mathcal{H}|=6$):
arg-max agreement $0.856$ against uniform allocation, mean label KL $0.185$,
step saving $1.1\%$ (`successive_halving.json`). The verdict is unsuccessful;
production labels stay uniform. Hierarchical selection is not implemented.

## 3. Deeper analysis of the high-load-perishable scenario

> *"Does DAHS's rule-selection distribution shift significantly under saturation?
> Could the minimum dwell constraint prevent timely adaptation in this regime?"*

**You named two candidate mechanisms with different remedies, and we now
distinguish them by measurement rather than attributing to "saturation" and moving
on.** Section 6.2 adds *"Boundary conditions: what the selector actually does under
saturation"*:

- **Does the selector stop selecting?** Measured as the **exponentiated entropy of
  the deployed-rule distribution**, which equals $|\mathcal{H}|$ when the selector
  spreads across the pool and 1.0 when it has collapsed onto one rule. If it
  collapses, the scenario KPI is really that rule's KPI and the method has
  degenerated into a static policy.
- **Does the guardrail bind?** Measured as the **blocked-switch rate** — the share
  of epochs at which the ranker's arg-max differed from the deployed rule *because
  the dwell was still active*. The switching controller now records every decision,
  so this is read off a run rather than inferred.

We also run the causal test your comment implies: a sweep of $T_{\min}$ **within
the high-load-perishable scenario alone**. If the cost-minimising dwell there is
shorter than the deployed one, the guardrail is the binding constraint, and the
honest response is to report the trade explicitly and consider a load-dependent
dwell rather than defend a fixed default.

On 1,600 epochs per scenario (`results/E13_saturation/scenario_behaviour.parquet`):

| Scenario | Eff. rules | Switch rate | Blocked-switch | Gate-open | EEDD share | COVERT share | $J$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| low load | 1.005 | 0.0006 | 0 | 0.931 | 0.999 | 0 | 9.82 |
| balanced | 1.034 | 0.0056 | 0 | 0.953 | 0.995 | 0.004 | 17.73 |
| default | 1.610 | 0.039 | 0.008 | 0.929 | 0.849 | 0.138 | 382.27 |
| high-load-perish | 1.584 | 0.031 | 0.0006 | 0.963 | 0.173 | 0.828 | 11430 |

Collapse onto EEDD is the light-load mechanism; a switch into COVERT is the
perishable-saturation mechanism. The dwell sweep *within* high-load-perishable
(`dwell_verdict_high_load_perish.json`) finds a two-way tie at the deployed
$T_{\min}=2$ and at $T_{\min}=3$ (both $11429.77$); $T_{\min}\in\{0,1\}$ is $11429.85$
and $T_{\min}=4$ is $11432.15$. The guardrail is not the boundary condition
there. WSPT, not DAHS, is the cheapest method in that scenario ($11171$ vs
$11430$).

## 4. Is the full feature set necessary?

> *"An additional ablation using only the top-5 features would help validate
> whether the full feature set is necessary."*

**Added** as the `top5_features` ablation in Section 6.8, selecting the five
highest-importance features by SHAP and retraining. We agree with the motivation:
a parsimonious selector is materially easier to deploy and to audit.

This reads alongside the redundancy analysis added for Reviewer 1's comment 3.a,
which found two features that were degenerate outright and removed them.

`top5_features` $J=383.99$ against DAHS $382.27$ (paired $+$1.72,
$[-3.44,6.59]$, $p_{\mathrm{adj}}=0.83$). Null on cost and SFR. The remaining
33 ranker coordinates are not earning their dimensionality on this 50-shift
test set. Train wall $897$ s.

## 5. Supplementary metrics for the ablations

> *"Current ablation only reports SLA breach rate; add training convergence speed
> and online inference latency."*

**Agreed, and this changed how we think about one of the components.** Every
ablation row now carries three column groups: composite cost and its decomposition
(does the component improve the objective?), **training wall-clock to convergence**
(what does it cost to fit?), and **per-decision inference latency, mean and p95**
(what does it cost to run?). Latency is measured around the policy call alone, with
the environment step excluded, so it is the controller's own cost.

Inference-only ablations record a null training wall-clock; that is the finding for
those rows rather than a gap.

The reason this mattered: the switching controller does *not* improve KPIs —
removing it improves them slightly — and it is retained deliberately as a
deployability guardrail. Reporting cost alongside benefit makes that an explicit
engineering trade rather than something to defend.

Manuscript Table 10 (six displayed rows; BH family is the eight methods in
`e3_summary.parquet`, including the omitted identity `random_ambiguity_filter`;
$p_{\mathrm{adj}}$ from that 8-arm correction):

| Ablation | $J$ | vs DAHS [95% CI] | $p_{\mathrm{adj}}$ | SFR | Train wall (s) |
|---|---:|---|---:|---:|---:|
| hard_labels | 383.28 | $+$1.01 [$-$5.52, 7.13] | 0.83 | 0.0665 | 1302 |
| no_calibration | 382.31 | $+$0.04 [$-$5.76, 4.89] | 0.83 | 0.0665 | --- |
| no_regime | 384.41 | $+$2.14 [0.04, 4.61] | 0.34 | 0.0671 | 1124 |
| no_switching_controller | 381.27 | $-$0.99 [$-$4.24, 1.96] | 0.83 | 0.0667 | --- |
| single_sample_rollout ($M{=}1$) | 381.08 | $-$1.18 [$-$7.83, 5.18] | 1.00 | 0.0664 | --- |
| DAHS | 382.27 | --- | --- | 0.0669 | --- |

The `top5_features` retrain ($J=383.99$; $+$1.72 [$-$3.44, 6.59]; SFR $0.0671$;
897 s) is no longer a row of that table. It selected its five coordinates from a
**test-set** SHAP ranking, so it is a diagnostic reported in the text of
Section 6.8, not a held-out ablation.

No ablation rejects equality with DAHS after BH control. Non-rejection is not
an equivalence test. Inference-only arms do not retrain. Mean decision latency
on the retrain arms is $0.7$--$0.9$ ms, below the deployed DAHS eval latency
in Table 13 ($4.24$ ms), which includes the switching wrapper and regime
posteriors.

## 6. Expand the limitations

> *"Expand Section 8 with detailed subsections covering simulation circularity,
> small heuristic pool, single warehouse setting, absence of online adaptive
> fine-tuning."*

**Done — Section 8 is restructured into the four subsections you name, plus a
fifth.**

- **8.1 Shared-simulator circularity.** The rollout labels and the evaluation use
  the same simulator. We give three mitigations and state that none dissolves the
  concern; what remains unaddressed is the transition function itself.
- **8.2 A small heuristic pool.** The ceiling is the pool's envelope — DAHS
  selects, it does not construct. Genetic programming is the established route to
  *generating* rules, and a hybrid is a natural combination we do not attempt.
- **8.3 A single warehouse setting.** One facility archetype; all robustness cells
  share the same simulator family.
- **8.4 No online adaptation after deployment.** The selector is fitted once and
  frozen; a warehouse drifts.
- **8.5 Reinforcement-learning baselines.** Added because a paper reporting a
  learned method beating RL owes evidence the baselines were configured competently
  — see Reviewer 1's comment 6.b.

We also added partial observability and its irreducible regret floor, and model
error, as numbered limitations at the head of the section.

---

# Reviewer 5

Your review is the one we found hardest to read and the most useful. If a reader
familiar with approximate dynamic programming could not determine what problem we
were solving, that is a failure of the paper and not of the reader. The changes
below are substantial and largely structural.

## 1. The problem definition is missing; the presentation is implementation-focused

> *"Frankly, I cannot state with certainty what the authors are doing. A problem
> definition is missing and the presentation of the method is very
> implementation-focused."*

**Accepted without qualification.** The submitted paper described the problem in
prose and moved directly to implementation. Two structural changes:

**New Section 3.2** states the problem as a sequential decision process before any
implementation detail (see item 3 below).

**New Section 1.1** defines every term and symbol in two tables, before first use.
This directly addresses your closing point, and we quote it back because it was the
most concrete diagnosis in your review: *"corpus of simulated shifts"*, *"held-out
shifts"*, *"SLA-breach rate"* and *"snapshot-trained ranker"* are each defined
there, along with shift, decision epoch, dispatching rule, selection
hyper-heuristic, rollout, truncated rollout, continuation, ranker, soft label,
regime, switching controller, ablation, and DAHS itself. The abstract is rewritten
in plain language and introduces no unexplained abbreviation.

## 2. How does this differ from value-function approximation or RL?

> *"I find it difficult to identify a clear difference to value function
> approximation or RL where values are also learned offline via simulation of a
> trained policy."*

**The submitted paper's answer was rhetorical, and the honest answer starts by
conceding your premise: the method is *inside* the ADP family, not outside it.**
What we describe is one step of approximate policy iteration in which the improved
policy is represented by a classifier rather than derived from a value function —
that is Rollout Classification Policy Iteration, and Section 2.4 now says so
(this is also Reviewer 1's comment 2.b).

Section 2.4 adds a subsection, *"How this differs from value-function
approximation"*, giving a *mechanism* statement rather than a novelty claim, in
three parts:

- **What is learned.** VFA fits $V(S)$ or $Q(S,u)$ — a scalar satisfying a Bellman
  fixed point approximately — and recovers a policy by one-step lookahead. Here no
  value function is formed and no fixed point is sought; the object fitted is the
  policy itself, trained on directly measured per-action costs.
- **How error behaves.** This is the difference that matters. A bootstrapped value
  estimate propagates error through the Bellman backup, so error at one state
  contaminates its predecessors. The classification construction has no backup: its
  error is ordinary supervised generalisation error and does not compound across
  states. **That is the mechanism our experiments isolate**, holding environment,
  corpus, model class, feature set and objective fixed — and it is why the paper is
  now framed as a controlled comparison of training signals.
- **What each requires.** This asymmetry runs against us and we state it plainly:
  VFA can be fitted from logged transitions alone, the data an operating warehouse
  already produces. Our construction needs a simulator that can be **reset to an
  arbitrary state and rolled forward under a counterfactual action**, because the
  label is the cost of rules that were *not* taken. That is a strictly stronger
  requirement, it is why the method is simulator-bound, and it makes the circularity
  of Section 8.1 intrinsic rather than incidental. A practitioner without a
  trustworthy simulator should prefer value learning from logs.

## 3. No model of the problem

> *"The authors do not present a model for their problem, e.g., a sequential
> decision process based on the framework of Warren Powell. What are the problem
> states, decisions, cost, etc.?"*

**New Section 3.2 does exactly this**, in Powell's canonical form and before any
implementation detail:

- **State** $S_t = (\mathcal{Q}_t, \mathbf{b}_t, t)$ — the multiset of waiting
  orders each carrying its full attribute tuple $(a_o, p_o, d_o, x_o, w_o)$; the
  vector recording when each picker next becomes free; and the clock.
- **Decision** $u_t \in \mathcal{H}$ — a rule from the pool.
- **Exogenous information** $W_{t+1}$ — the orders arriving in $(t, t+1]$ with
  their attributes.
- **Transition** $S_{t+1} = S^M(S_t, u_t, W_{t+1})$ — the admission–rank–assign
  procedure, written out explicitly in Section 3.4, including the admission rule.
- **Objective**
  $\min_{\pi} \mathbb{E}\big[\sum_t C(S_t, U^\pi_t(S_t), W_{t+1})\big]$.
- **Policy class** — a policy-function approximation
  $U^\pi(S_t) = \arg\max_h f_\theta(\phi(S_t))_h$.

We then state that $\phi(S_t)$ is an observation rather than a sufficient
statistic, so the problem is a POMDP (Reviewer 2's comment 4).

## 4. The relevant literature is not cited

> *"...e.g., the works by Matthias Klapp or Marlin Ulmer... the group proposed
> integrating RL in the simulation of a rollout, truncating the rollout horizon and
> approximating the remainder via RL, or using RL to determine the horizon
> state-dependently. None of the works are cited. Notably, also no C&OR-paper is
> cited."*

**Added, in a new Section 2.4 subsection on rollout and ADP for dynamic
dispatching.** Klapp et al. on the dispatch-waves problem — a neighbouring
wave-release decision, not the inner rule-selection decision studied here;
Goodson et al.'s rollout framework for finite-horizon stochastic dynamic programs,
including the treatment of truncated horizons that Sketch 1 sits inside;
Ulmer et al. on modelling conventions for stochastic dynamic routing and
dispatching (which Section 3 now follows), on offline–online ADP, on budgeting
decision time, and on anticipation against reactive re-optimisation.

**One thread deserves particular emphasis because it supersedes part of our
construction, and we say so rather than cite it in passing.** That literature does
not simply truncate a rollout and discard the tail: it **approximates the remainder
with a learned value function**, and in places uses learning to set the horizon
state-dependently. Our Sketches 1 and 2 treat truncation as a hard cut, which
makes the truncation bias $(H_t - \tau)\bar{C}$ a quantity to be tolerated rather
than estimated. A value-approximated tail would replace that with an approximation
error that need not grow with the remaining horizon — strictly the better
construction, and it would likely permit a shorter $\tau$, which is attractive here
because Sketch 2 says short horizons also limit model-error accumulation. We
do not implement it, and Section 9 records it as the most promising extension
rather than as an incidental idea.

**On the C&OR remark:** reference 3 of the submitted version (Mahmoudinazlou et
al., 2025) is a *Computers & Operations Research* paper, so the statement as
written is not quite right. We take the underlying point — that we had not engaged
with the C&OR and ADP dynamic-dispatching literature — as correct, and the
additions above are our response to it.

## 5. The writing is too dense and unspecific

> *"The abstract is already a good example. No abbreviations are introduced. What
> is a 'corpus of simulated shifts'? What are 'held-out shifts'?..."*

**The abstract is rewritten** in plain language, introducing no unexplained
abbreviation and stating the problem, the comparison and the two bounds without
jargon. **Section 1.1** defines every term and symbol, as described under item 1
above, and each of the four terms you name appears there by name.

---

# Points on which we did not do what was asked

We list these separately rather than leave them to be discovered.

1. **Reviewer 3, comment 2 — hierarchical selection.** Described and motivated in
   Section 6.12, not implemented. With six screened rules the flat selector is not
   the bottleneck. We record it as future work rather than claim it.

2. **Reviewer 5, comment 4 — value-approximated rollout tails.** We cite the
   Ulmer-line construction, state that it supersedes our hard-truncation treatment,
   and do not implement it. Section 9 records it as the most promising extension.

3. **Reviewer 1, comment 4.d — the pool could expand further.** We screened nine
   candidates rather than the larger set the comment contemplates. Given that one
   rule already wins 65% of decisions, we judged that the binding constraint is the
   *diversity* of the pool rather than its size, and report that finding instead.
   A GP-generated candidate set is recorded as future work in Section 8.2.

4. **Reviewer 2, comment 6 — the truncated question list.** As noted above, the
   copy of the review we received ends mid-sentence. We answered the four questions
   we inferred and would welcome the complete text.

---
