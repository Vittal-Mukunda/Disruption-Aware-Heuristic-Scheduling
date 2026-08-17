"""Final audit: every reviewer item -> a grep-checkable anchor in code and paper."""
import pathlib
import re

_RAW = pathlib.Path("paper/manuscript.md").read_text(encoding="utf-8")
# Strip YAML frontmatter before any citation scan: author emails contain
# '@', and an affiliation domain is not a bibliography key.
MS = _RAW
MS_BODY = _RAW
if _RAW.startswith("---"):
    _marker = chr(10) + "---"
    _end = _RAW.index(_marker, 3) + len(_marker)
    MS_BODY = _RAW[_end:]
BIB = pathlib.Path("paper/references.bib").read_text(encoding="utf-8")


def code(*paths):
    out = ""
    for p in paths:
        fp = pathlib.Path(p)
        if fp.exists():
            out += fp.read_text(encoding="utf-8", errors="replace")
        else:
            return None
    return out


# (item, paper anchors (all must appear), code paths (all must exist), code anchors)
CHECKS = [
 ("R1.1a routes/batching/travel exogenous",
  ["Scope: which order-picking decision", "Routing and travel-time estimation** are exogenous"], [], []),
 ("R1.1b warehouse lit review expanded",
  ["## 2.2 Dispatching rules and data-centric control in warehousing", "boysen2019warehousing"], [], []),
 ("R1.1c priority + perishability in objective",
  ["Priority class now enters the objective", "Perishability now enters the objective"],
  ["simulation/cost.py"], ["use_priority_weights", "w_spoil"]),
 ("R1.1d perishability binds? measured",
  ["## 3.5 Does perishability bind", "expiry-pivotal"],
  ["experiments/perishability_diagnostic.py"], ["pivotal"]),
 ("R1.1e order-level expiry justified",
  ["Where the expiry of an order comes from", "after* lot allocation", "min_{\\ell \\in o} x_\\ell"], [], []),
 ("R1.1f FEFO != EDD",
  ["FEFO is not a due-date rule", "That is EDD"],
  ["simulation/heuristics.py"], ["expiry_time"]),
 ("R1.2a/e positioning retired",
  ["## 2.6 Positioning and what this paper contributes",
   "the mechanism at the centre of this paper is not\nnovel", "We withdraw those claims"], [], []),
 ("R1.2b RCPI cited", ["lagoudakis2003rcpi", "fern2006api"], [], []),
 ("R1.2c multi-pass cited", ["wu1988multipass", "mouelhi2010neural", "shiue2020rl"], [], []),
 ("R1.2d Durasevic mischaracterisation corrected",
  ["We correct a characterisation", "Genetic programming in this literature is used predominantly to *generate*"], [], []),
 ("R1.3a feature provenance + correlation/VIF",
  ["Appendix A. State features, their provenance", "## A.3 Redundancy analysis", "variance inflation"],
  ["experiments/feature_analysis.py"], ["FEATURE_PROVENANCE", "vif"]),
 ("R1.3b 1600 test states explained", ["1600$ states before\nfiltering"], [], []),
 ("R1.4a/b pool motivated; FIFO justified",
  ["## 3.6 The rule pool", "zero-information control"], [], []),
 ("R1.4c ATC calibrated twice",
  ["ATC is calibrated, not assumed", "k^\\star_\\text{standalone}",
   "k^\\star_\\text{portfolio}", "This settles the WSPT/ATC inversion"],
  ["experiments/calibrate_rules.py"], ["portfolio", "standalone"]),
 ("R1.4d pool expanded + screening table",
  ["Screening is by marginal contribution", "COVERT", "marginal contribution | 95% CI"],
  ["experiments/calibrate_rules.py"], ["screen"]),
 ("R1.4e complementarity over state space",
  ["grid of the two\nstate dimensions", "diversity_state_grid"],
  ["experiments/calibrate_rules.py"], ["diversity"]),
 ("R1.5a/c parameter provenance + triangular disclosed",
  ["Parameters and their provenance", "Triangular$(2, 5, 12)$", "Triangular$(15, 45, 90)$"], [], []),
 ("R1.5b fit to real data",
  ["Fitting, not validating", "Appendix C. Fitting the input distributions"],
  ["experiments/fit_input_distributions.py"], []),
 ("R1.6a WSPT/FIFO anomalies explained",
  ["Two anomalies in the submitted results", "Cause 1", "Cause 2"],
  ["simulation/warehouse_env.py"], []),
 ("R1.6b RL sensitivity + coverage",
  ["Sensitivity analysis", "gap_closed_fraction", "Action coverage, and a correction"],
  ["experiments/rl_sensitivity.py"], ["coverage"]),
 ("R1.6c composite cost primary",
  ["is the primary metric of comparison", "Rank the table by composite cost"],
  ["experiments/e4_sensitivity.py"], ["weights"]),
 ("R1.7a DAHS defined", ["**DAHS** | Disruption-Aware Heuristic Scheduling"], [], []),
 ("R1.7d no reviewer-addressing in body", [], [], []),  # checked separately
 ("R1.7e reference 2 complete", ["dokeroglu2024hyperheuristics"], [], []),
 ("R2.1 unserved charged + breach formula stated",
  ["How the reported rates are calculated", "breach rate}_{\\text{arrived}}", "f_o = T + p_o"],
  ["simulation/kpis.py"], ["service_failure"]),
 ("R2.2 spoilage mechanics",
  ["Spoilage mechanics, stated explicitly", "Is a spoiled order counted in the breach count"],
  ["simulation/orders.py"], ["expiry_time"]),
 ("R2.3 multi-sample rollouts + variance",
  ["Why $M > 1$", "\\widehat{\\mathrm{se}}_h", "Common random numbers"],
  ["labeling/rollout_labeler.py"], ["stderr", "se_"]),
 ("R2.4 POMDP + aliasing witness",
  ["The observation is not the state", "partially observed** Markov decision process", "**A witness**"],
  ["experiments/observability_analysis.py"], []),
 ("R2.5 model misspecification",
  ["## 6.11 Model misspecification", "Proposition 2"],
  ["experiments/misspecification.py"], []),
 ("R2.6 online rollout MPC baseline",
  ["rolling-horizon"], ["baselines/rolling_horizon_mpc.py"], []),
 ("R3.1 offline compute cost",
  ["### Offline cost", "interval-steps"],
  ["experiments/compute_budget.py"], []),
 ("R3.2 pool-size scalability",
  ["### Scaling in $|\\mathcal{H}|$", "Adaptive sample allocation", "Hierarchical selection"],
  ["labeling/rollout_labeler.py"], ["successive_halving"]),
 ("R3.3 saturation deep-dive",
  ["Boundary conditions: what the selector actually does under saturation", "blocked-switch rate"],
  ["experiments/saturation_analysis.py"], []),
 ("R3.4 top5 feature ablation",
  ["`top5_features`"], ["experiments/e3_ablations.py"], ["top5_features"]),
 ("R3.5 ablation train time + latency",
  ["training wall-clock to convergence", "per-decision inference latency"],
  ["experiments/e3_ablations.py"], ["wall_clock"]),
 ("R3.6 limitations expanded",
  ["### 8.1 Shared-simulator circularity", "### 8.2 A small heuristic pool",
   "### 8.3 A single warehouse setting", "### 8.4 No online adaptation after deployment"], [], []),
 ("R5.1/5.3 sequential decision model (Powell)",
  ["**The sequential decision process**", "powell2019unified", "S^M(S_t, u_t, W_{t+1})"], [], []),
 ("R5.2 differs from VFA/RL",
  ["### How this differs from value-function approximation"], [], []),
 ("R5.3 Klapp / Ulmer / C&OR cited",
  ["klapp2018onedim", "ulmer2020modeling", "goodson2017rolloutframework"], [], []),
 ("R5.4 terminology defined",
  ["## 1.1 Terminology and notation", "corpus of simulated shifts", "snapshot-trained ranker"], [], []),
]

fails = []
for item, paper_anchors, code_paths, code_anchors in CHECKS:
    for a in paper_anchors:
        if a not in MS and a not in BIB:
            fails.append(f"{item}: PAPER missing {a!r}")
    src = code(*code_paths) if code_paths else ""
    if src is None:
        fails.append(f"{item}: CODE file missing {code_paths}")
        continue
    for a in code_anchors:
        if a not in src:
            fails.append(f"{item}: CODE missing {a!r} in {code_paths}")

# R1.7c: paragraph titles ending '..'
if re.search(r"\*\*[^*\n]*\.\.\*\*", MS):
    fails.append("R1.7c: a bold paragraph title still ends in '..'")

# R1.7d: the ARTICLE must not address reviewers. Editorial scaffolding may:
# HTML comment blocks (draft notes), blockquote revision notes, and TBD-rerun
# spans are all removed before submission, so they are exempt. What is left is
# prose a referee would actually read.
_prose = re.sub(r"<!--.*?-->", "", MS_BODY, flags=re.S)
_prose = re.sub(r"⟨TBD-rerun.*?⟩", "", _prose, flags=re.S)
_prose = chr(10).join(
    ln for ln in _prose.splitlines() if not ln.lstrip().startswith(">")
)
for m in re.finditer(r"[Rr]eviewer", _prose):
    _s = _prose.rfind(chr(10), 0, m.start()) + 1
    _e = _prose.find(chr(10), m.start())
    fails.append(
        "R1.7d: reviewer-addressing in article prose: "
        + _prose[_s:_e].strip()[:70]
    )

# R1.7b: the duplicated contributions paragraph
if MS.count("tempered-softmax label, ablation") > 0:
    fails.append("R1.7b: duplicated P3 paragraph still present")

# Bibliography closure
keys = set(re.findall(r"@\w+\{([^,]+),", BIB))
used = {k.rstrip(".,;:") for k in re.findall(r"@([A-Za-z][A-Za-z0-9_:.+-]*)", MS_BODY)}
if used - keys:
    fails.append(f"BIB: cited but missing: {sorted(used - keys)}")
if keys - used:
    fails.append(f"BIB: present but uncited: {sorted(keys - used)}")

print(f"{len(CHECKS)} reviewer items checked")
if fails:
    print(f"\n{len(fails)} FAILURES:")
    for f in fails:
        print("  -", f)
else:
    print("\nALL CHECKS PASS")
