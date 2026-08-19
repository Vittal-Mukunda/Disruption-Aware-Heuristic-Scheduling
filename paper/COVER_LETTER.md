# Cover letter — resubmission of CAOR-D-26-01812

To the Editor,
*Computers & Operations Research*

Dear Editor,

Please find enclosed a revised version of manuscript **CAOR-D-26-01812**,
*"Offline Rollout Distillation for Warehouse Order Dispatching: A Controlled
Comparison of Training Signals"*, together with a point-by-point response to the
four reviewers. (The submitted title claimed sample-efficient adaptive selection;
we have retitled the paper to match what the campaign actually supports.)

We are grateful for reviews that were unusually detailed and technically precise.
Several comments identified defects in our implementation rather than only in our
exposition, and we want to be direct with you about the scale of what followed.

**The revision is substantial and every quantitative result has been
regenerated.** Four of the comments each independently invalidated the numbers in
the submitted version: the objective discounted an abandoned order by a factor of
600 against one served late and the reported breach rate excluded abandoned orders
from its denominator (Reviewer 2); the rollout labels were single-path
realisations rather than estimates of expected cost (Reviewer 2); the ATC rule was
never calibrated and the rule pool was neither motivated nor screened (Reviewer 1);
and perishability entered neither a constraint nor the cost, while the rule we
called FEFO was in fact EDD (Reviewers 1 and 2). Each of these changes the
data-generating process or the objective, so we rebuilt the model and re-ran the
entire experimental campaign. No number from the submitted version is carried
forward into a claim.

**We have withdrawn the paper's novelty claim.** Reviewers 1 and 5 are correct
that simulating a rule pool offline and fitting a classifier to the result is not
new — it is rollout classification policy iteration in the reinforcement-learning
literature and multi-pass rule selection in the scheduling literature. Section 2
now places the method inside both traditions, corrects our mischaracterisation of
the dispatching-rule-selection literature, and engages with the Klapp and Ulmer
line of work on rollout and approximate dynamic programming for dynamic
dispatching that we had not cited. The contribution is reframed as an empirical
study — a controlled comparison of training signals — and we state explicitly
that we make no claim to a new training mechanism. Propositions 1 and 2 are
truncation and model-error sketches for the labels; they are not confirmed by
deployed cost.

**Several of the new results are less favourable to the method than the submitted
ones, and we report them as such.** Charging unserved orders on the submitted
event logs compressed the FIFO margin from about 3.8× to about 1.2×; after
regenerating under causal admission the live FIFO gap is **3.90×** on composite
cost (Table 6). Online truncated lookahead remains cheaper than the distilled
selector (greedy $J=356$, rolling $J=363$, DAHS $J=381$). Labels from $M=1$
through $M=40$ and horizons $\tau=1$ through $\tau=4$ sit in a null band on
deployed $J$. WSPT, screened out of the default pool, beats DAHS under
high-load-perishable. On the 12-cell robustness grid DAHS wins 0 of 12 cells
among the four frozen methods; the one-step teacher wins 8. We have rewritten
Sections 6.2, 6.4, 6.5 and 7 around those outcomes rather than the anticipated
ones.

**One request.** Reviewer 2's sixth comment ends mid-sentence in the copy we
received — *"Including this benchmark would answer several critical questions:"*
— with the list of questions truncated. We have implemented the benchmark and
answered the four questions we believe were intended, and marked them in our
response as our reading. If the complete comment can be supplied we would be glad
to address the specific questions.

All authors have approved the revised manuscript. The work is original, is not
under consideration elsewhere, and we declare no competing interests. The code,
data-generation pipeline and all result artifacts are available at
https://github.com/Vittal-Mukunda/Disruption-Aware-Heuristic-Scheduling.

We hope the revision addresses the reviewers' concerns, and we thank them for the
care they took over the original submission.

Yours sincerely,

**Vittal Mukunda**, Atharva Somani, Pranjal Malaiya
Department of Industrial Engineering and Management
R. V. College of Engineering, Bengaluru, India
vittalmukunda.im24@rvce.edu.in

19 August 2026
