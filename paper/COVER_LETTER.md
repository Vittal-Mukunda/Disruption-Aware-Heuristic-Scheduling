# Cover letter — resubmission of CAOR-D-26-01812

> **DRAFT.** Fill the ⟨…⟩ slots and delete this box before sending. Keep it to
> one page; the detail belongs in the response document.

---

To the Editor,
*Computers & Operations Research*

Dear Professor ⟨editor name⟩,

Please find enclosed a revised version of manuscript **CAOR-D-26-01812**,
*"Sample-Efficient Adaptive Heuristic Selection via Offline Rollout Distillation
for Dynamic Warehouse Order Dispatching"*, together with a point-by-point response
to the four reviewers.

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
study — a controlled comparison of training signals at matched data budgets — and
we state explicitly that we make no claim to a new training mechanism.

**Several of the new results are less favourable to the method than the submitted
ones, and we report them as such.** Charging orders that are never served
compresses our advantage over the simplest baseline from roughly 3.8× to roughly
1.2×; we state this in Section 6.2 before reporting any new number. The
state-space complementarity analysis Reviewer 1 asked for shows one rule winning
65% of decisions and owning 15 of 16 cells of the state-space grid, which leaves a
selector far less room than the submitted four-rule pool appeared to; we report the
oracle gap and the fact that it exceeds our own pre-registered concentration
ceiling. We judged that reporting these plainly was more useful than
re-engineering the study around them.

⟨IF THE CAMPAIGN RESULT IS UNFAVOURABLE — adapt or delete: We should also draw
your attention to ⟨finding⟩, which materially weakens ⟨claim⟩ relative to the
submitted version. We have rewritten Section ⟨n⟩ around the measured outcome rather
than the anticipated one.⟩

**One request.** Reviewer 2's sixth comment ends mid-sentence in the copy we
received — *"Including this benchmark would answer several critical questions:"* —
with the list of questions truncated. We have implemented the benchmark and
answered the four questions we believe were intended, and marked them in our
response as our reading. If the complete comment can be supplied we would be glad
to address the specific questions.

All authors have approved the revised manuscript. The work is original, is not
under consideration elsewhere, and we declare no competing interests. The code,
data-generation pipeline and all result artifacts are available at
⟨repository URL⟩ ⟨or: will be made available on acceptance⟩.

We hope the revision addresses the reviewers' concerns, and we thank them for the
care they took over the original submission.

Yours sincerely,

**Vittal Mukunda**, Atharva Somani, Pranjal Malaiya
Department of Industrial Engineering and Management
R. V. College of Engineering, Bengaluru, India
vittalmukunda.im24@rvce.edu.in

⟨date⟩
