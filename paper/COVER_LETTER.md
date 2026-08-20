# Cover letter — resubmission of CAOR-D-26-01812

To the Editor,
*Computers & Operations Research*

Dear Editor,

Please find enclosed a revised version of manuscript **CAOR-D-26-01812**,
*"Offline truncated-rollout labels did not recover the online teacher on one
warehouse simulator"*, together with a point-by-point
response to the four reviewers.

The previous titles claimed a new selector, then a generalisation about
training signals. This version titles the result on the simulator the campaign
actually ran. The article is recast as a standalone negative-result study, not as
a diff against the original submission. Table 5 of the last revision (the
superseded scoreboard) is gone, as are the truncation sketches and the
changelog appendix. The confirmatory functional is mean composite cost, because
a warehouse pays total cost; Wilcoxon signed-rank is secondary on every claim.
The `top5_features` retrain is out of Table 10. The PPO grid remains a
test-scored diagnostic; Table 5 keeps `ppo_fair`.

**Every quantitative result has been regenerated.** Live tables use causal
periodic-review admission with a terminal admit at shift end (mean arrived
$|A|=791$). Production labels were generated with that flag off and were not
regenerated.

**The mechanism is not new**, and the paper does not claim it. Simulating a
rule pool offline and fitting a classifier is rollout classification policy
iteration and multi-pass rule selection. Klapp et al. is a neighbouring
wave-release decision, not the inner rule-selection decision studied here.

**The distilled ranker does not recover the teacher.** After regeneration,
online truncated lookahead remains cheaper (greedy $J=357$, rolling $J=363$,
DAHS $J=382$). Always-COVERT is the static rule to beat ($J=455$; 49/50
shifts). FIFO's live gap is **3.89×** on composite cost (SFR $0.1837$). A
one-step snapshot matches DAHS on the confirmatory paired interval for $J$. A
two-step label is cheaper than the deployed four-step ranker on the mean
($-9.11$, $[-17.61,-1.35]$), with 21 wins, 20 losses, 9 ties and median
difference 0. On the 12-cell robustness grid five methods are frozen: DAHS
wins 0 of 12; the one-step teacher wins 8; Always-EEDD wins the four light
cells. Distillation is 4.24 ms versus 670 ms of lookahead. At a 15-minute
review, 670 ms is **0.07%** of the epoch. If a resettable simulator exists,
run the teacher. If it does not, these labels cannot be generated.

**One request.** Reviewer 2's sixth comment ends mid-sentence in the copy we
received — *"Including this benchmark would answer several critical
questions:"* — with the list of questions truncated. We have implemented the
teacher and answered the four questions we believe were intended.

All authors have approved the revised manuscript. The work is original, is
not under consideration elsewhere, and we declare no competing interests.
The code, data-generation pipeline and all result artifacts are available at
https://github.com/Vittal-Mukunda/Disruption-Aware-Heuristic-Scheduling.

Yours sincerely,

**Vittal Mukunda**, Atharva Somani, Pranjal Malaiya
Department of Industrial Engineering and Management
R. V. College of Engineering, Bengaluru, India
vittalmukunda.im24@rvce.edu.in

20 August 2026
