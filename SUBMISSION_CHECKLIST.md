# Submission checklist — CAOR-D-26-01812

Everything between "the campaign finished" and "submitted". Ordered so that the
things which could still change the paper come before the things which only
format it.

`[ ]` unchecked · `[~]` in progress · `[x]` done

---

## 0. Before you start the campaign

- [x] `scripts/preflight.py` passes
- [x] `pytest` green (94 passed, 11 skipped — every skip names a missing campaign
      artifact or the FEFO-not-in-pool no-op)
- [x] `scripts/audit_reviewer_items.py` prints ALL CHECKS PASS (40/40)
- [x] `config.yaml` carries the Stage-1 pool `[EEDD, COVERT, MS, ATC, MDD, EDD]`
      and fitted scales `3.0 / 4.0`
- [x] `requirements-lock.txt` present — install from it, not the pyproject ranges
- [ ] **Ask the editor for the complete text of Reviewer 2's comment 6.** It ends
      mid-sentence in our copy. This is the one input we are missing and it does
      not depend on the run. Do it now, not after.
- [ ] Decide the R1.2e repositioning formally (see §3 below). The paper currently
      commits to "controlled study of training signals"; if you want a different
      framing, changing it after the results are written in is more work.

## 1. Run the campaign

- [ ] Follow `RUN_PROMPT.md` on the run machine
- [ ] `CAMPAIGN_REPORT.md` produced and committed
- [ ] Every artifact in `RUN_CAMPAIGN.md` §3 exists
- [ ] `pytest` re-run *after* the campaign — the 11 skips should now be 0 or close
      to it. Any test that now runs and **fails** is a result problem, not a test
      problem, and must be understood before anything is written.

## 2. Read the results before writing anything

These are the four places the campaign could tell you the paper needs
restructuring rather than rewriting. Check them first, in this order.

- [ ] **Does DAHS beat EEDD-alone on composite cost, with a paired interval
      excluding zero?** EEDD wins 65% of decisions and owns 15/16 state-space
      cells; the win-rate oracle gap is 7.29 points. If DAHS does not clear EEDD
      meaningfully, the "selection beats any single rule" framing does not survive
      and §6.2, §7 and the abstract need rebuilding around sample efficiency and
      amortisation instead.
- [ ] **`frac_separation_below_1se` in `data/label_meta.json`.** 50.4% on a
      4-shift smoke corpus with the deployed six-rule pool (76.8% on the nine-rule
      candidate set). If it stays near half at full scale, the rollout does not
      resolve half the decisions — report it, and make the $M$ sweep the headline
      of §6.4 rather than a supplementary result.
- [ ] **`gap_closed_fraction` for PPO, and the FQI coverage fix.** §6.9 and §6.10
      are written conditionally. If either gap closes materially, withdraw the
      structural claim and adopt the tuned configuration as the baseline throughout.
      Do not keep the unfavourable branch and the favourable numbers.
- [ ] **Does the calibrated E8 grid cell reproduce the Table 1 static rows
      exactly?** `tests/test_reproducibility.py` pins this. If it fails, stop —
      the submitted version had a contradiction here and nothing downstream is
      trustworthy until it is resolved.

## 3. Decisions that are yours, not the data's

- [ ] **R1.2e repositioning.** The paper commits to (b) *controlled study of
      training signals*, with elements of (a) *application and system integration*.
      Confirm or change. If the §2 result above is unfavourable, (b) becomes the
      only defensible framing, since it is a *relative* claim at matched budgets and
      survives a small absolute margin.
- [ ] **Author order** — currently Mukunda, Somani, Malaiya, matching the IEEE
      version. Confirm with your co-authors.
- [ ] **Repository disclosure** — decide whether the GitHub URL goes in the paper,
      the cover letter, or is withheld until acceptance. The cover letter has a slot.
- [ ] **Whether to strip the Claude co-author trailer from the 17 already-pushed
      commits.** Cosmetic, requires a force-push, breaks any existing clone. Only
      matters if the repository is disclosed.

## 4. Write the paper

- [ ] Fill all **44 `⟨TBD-rerun⟩` markers** in `paper/manuscript.md`.
      `grep -c "TBD-rerun" paper/manuscript.md` must reach 0.
      Each marker states what to report *and* which way the conclusion falls —
      resolve against the measured outcome, do not delete the unfavourable branch.
- [ ] Regenerate every figure; confirm all nine referenced paths exist
- [ ] Re-check the abstract against the final numbers. It is written last for a
      reason: it currently carries a `⟨TBD-rerun⟩` for the headline findings.
- [ ] Re-read §7 Discussion and §9 Conclusion end to end against the final results.
      These are the two sections most likely to retain an optimistic framing that
      the numbers no longer support.
- [ ] Remove the draft scaffolding: the *"Revision note on pending numbers"*
      blockquote at the top, and the `DRAFT v2` / `DRAFT v3` HTML comments
- [ ] `scripts/audit_reviewer_items.py` still passes after all edits

## 5. Fill the response letter

- [ ] Fill every `⟨PENDING⟩` in `paper/RESPONSE_TO_REVIEWERS.md` from the artifact
      named beside it (index at the foot of that file)
- [ ] **Verify every reviewer quotation against the decision letter.** They are
      abridged for readability and must not misrepresent.
- [ ] Check the *"Points on which we did not do what was asked"* section is still
      accurate — items may have moved in or out during the run
- [ ] Re-read for tone. It should read as a colleague reporting findings, not as a
      defendant. Where a reviewer was right, say so plainly and once.

## 6. Format and package

- [ ] `python scripts/build_submission.py --check` — reports what is missing
- [ ] Convert to Elsevier `elsarticle` (needs pandoc + a LaTeX toolchain; neither
      is installed on the dev laptop). `scripts/build_submission.py` drives it.
- [ ] Figures at journal resolution, in the required format, each referenced in
      text and numbered in order of first mention
- [ ] Tables numbered in order; every table referenced in text
- [ ] References: check the `.bbl` against `references.bib`; confirm every DOI
      resolves; Elsevier numbered style
- [ ] Highlights (3–5 bullets, ≤85 characters each) — **must be rewritten against
      the final numbers**, not the submitted ones
- [ ] Declaration of competing interests
- [ ] CRediT author-contribution statement for all three authors
- [ ] Data availability statement
- [ ] Word/page count against the journal limit
- [ ] Fill `paper/COVER_LETTER.md` slots: editor name, repository URL, date, and
      the unfavourable-finding paragraph if it applies

## 7. Final read

- [ ] Read the whole manuscript once, start to finish, on paper or in PDF — not in
      the editor. Structural problems only surface this way.
- [ ] Confirm no superseded number survives as a live claim.
      `grep -n "0.0133\|1.33%\|3.09\|2.40 points"` should only match inside tables
      explicitly labelled **(superseded)**.
- [ ] Confirm the paper addresses no reviewer anywhere in its prose (the audit
      checks this, but read for it too)
- [ ] Confirm the abstract, §1 contributions, §7 and §9 tell the *same* story as
      §6 — the one the data supports, not the one we set out to tell
- [ ] Co-authors have read and approved

---

## What is genuinely uncertain

Recorded here so it is not rediscovered under time pressure.

**The paper may need restructuring rather than rewriting.** Three of the four
checks in §2 above have a plausible outcome that removes a headline claim. That
is a consequence of fixing what the reviewers correctly identified: the submitted
margins were partly artefacts of an objective that did not charge for abandoned
orders, an uncalibrated ATC, and a dispatcher that idled pickers for
arrival-agnostic rules. A smaller, honest result is the expected outcome and is
still publishable — the training-signal comparison is a *relative* claim at
matched budgets and does not depend on the absolute margin.

**The run is ~16 h base, 20–25 h on a laptop**, and large parts of the pipeline
have never executed end to end. `preflight.py` checks that modules import, which
is not the same thing: the R2.4 aliasing witness imported cleanly for the entire
project and was silently reporting a zero cost gap for a claim it could not
support. Budget time for at least one stage failing and needing a fix.
