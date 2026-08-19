# Submission checklist — CAOR-D-26-01812

Everything between "the campaign finished" and "submitted". Ordered so that the
things which could still change the paper come before the things which only
format it.

`[ ]` unchecked · `[~]` in progress · `[x]` done

---

## 0. Before you start the campaign

- [x] `scripts/preflight.py` passes
- [x] `pytest` green on the campaign machine
- [x] `scripts/audit_reviewer_items.py` prints ALL CHECKS PASS (40/40) on the
      current manuscript (re-run after every prose edit)
- [x] `config.yaml` carries the Stage-1 pool `[EEDD, COVERT, MS, ATC, MDD, EDD]`
      and fitted scales `3.0 / 4.0`
- [x] `requirements-lock.txt` present — install from it. Python **3.12 only**
      (`requires-python = ">=3.12,<3.13"`)
- [ ] **Ask the editor for the complete text of Reviewer 2's comment 6.** It ends
      mid-sentence in our copy. This is the one input we are missing and it does
      not depend on the run. Do it now, not after.

## 1. Campaign (done) and leftover completeness evals (not done)

- [x] Revision campaign + M-sweep + `compute_budget measure`/`scaling` on `C:\CAOR`
- [x] `CAMPAIGN_REPORT.md` produced (`ccf0240`) and later manuscript rewrite
- [x] Live Table 6 matches `results/E2/default_stats.parquet` at `terminal_admit: false`
      (mean `|A|=767`)
- [ ] **Leftover completeness evals** — follow current `RUN_PROMPT.md` on `C:\CAOR`
      only. All of A–G are mandatory. Do not run them on a OneDrive clone.
      - A terminal admit (`sim.terminal_admit: true`) then re-eval Table 6
      - A2 scenario Table 7 (including high_load_perish WSPT)
      - B ATC k=1.5 in-memory overlay into `results/E_atc_k1p5`
      - C E8 + Always-COVERT
      - D teachers M=20 into `results/E_teacher_M20`
      - E PPO HP selected on calib, frozen on test
      - F eval-only refresh of E3/E4/E10/E13/A2/data-efficiency
      - G `python -m experiments.compute_budget latency`

## 2. Read the results before writing anything

- [x] **Does DAHS beat EEDD-alone on composite cost, with a paired interval
      excluding zero?** Yes on cost (381 vs 696). The static to beat is
      **Always-COVERT** (454), not EEDD. One-step lookahead (356) and the τ=4
      teacher (363) both beat DAHS. Rebuild §7 around amortisation + training
      signal, not "selection beats any single rule vs the win-rate champion".
- [x] **`frac_separation_below_1se` in `data/label_meta.json`.** 33.4% at full
      scale (M=20, |H|=6). Report it. M-sweep is complete (M in {1,5,10,20,40}).
- [x] **`gap_closed_fraction` for PPO, and the FQI coverage fix.** PPO closed
      78.3% on a **test-scored** grid — that cell is not Table 6. Table 6 keeps
      untuned `ppo_fair` (611). FQI coverage under `random` is adequate.
      DAHS beats FQI 381 vs 397 (1.04x, CI excludes zero).
- [x] **Does the calibrated E8 grid cell reproduce the Table 6 static rows
      exactly?** Yes for the four frozen methods. Always-COVERT is still absent
      until leftover C.

## 3. Decisions that are yours, not the data's

- [x] **R1.2e repositioning.** The paper commits to a controlled study of
      training signals. Title and three contributions match that.
- [ ] **Author order** — currently Mukunda, Somani, Malaiya. Confirm with
      co-authors. Do not change YAML without that confirmation.
- [x] **Repository disclosure** — GitHub URL is in the cover letter.
- [ ] **Whether to strip the Claude co-author trailer from already-pushed
      commits.** Cosmetic, requires a force-push. Only matters if the repository
      is treated as the archival record.

## 4. Write the paper

- [x] Fill all `⟨TBD-rerun⟩` markers (count is 0 on the current manuscript)
- [x] Abstract rewritten last against live numbers (qualify the $450 PPO cell)
- [x] Draft scaffolding removed
- [x] `scripts/audit_reviewer_items.py` 40/40 after the rewrite
- [ ] After leftover A–G: replace every number those evals change, including
      mean `|A|`, Table 6–7, E8, teachers M=20, calib PPO, latency, E3/E4/E10/E13/A2/DE
- [ ] Re-read §7 and §9 against the post-admit results

## 5. Fill the response letter

- [x] No `⟨PENDING⟩` / `⟨…⟩` slots
- [ ] After leftover A–G: refresh live Table 6 numbers in the response
- [ ] Check the *"Points on which we did not do what was asked"* section is
      still accurate
- [x] Hierarchical selection is stated as not implemented
- [x] 1.20× is the old-log diagnosis; 3.90× is live Table 6 (pre-admit)

## 6. Format and package

- [x] `python scripts/build_submission.py --check` READY on the current tree
      (re-run after leftover-number edits)
- [ ] Convert to Elsevier `elsarticle` (needs pandoc + a LaTeX toolchain)
- [ ] Figures at journal resolution; numbering already consecutive 1–11 / tables 1–14
- [x] Highlights 5 bullets, each ≤85 characters, rewritten against live numbers
- [x] Cover letter filled (editor as "Editor", repo URL, 19 August 2026)
- [ ] Word/page count against the journal limit
- [ ] CRediT / competing interests / data availability as the journal requires

## 7. Final read

- [ ] Read the whole manuscript once, start to finish, on paper or in PDF
- [ ] Confirm no superseded number survives as a live claim
      (`0.0133`, `1.33%`, `3.09`, `2.40 points` only inside tables labelled
      **(superseded)**)
- [ ] Confirm the paper addresses no reviewer in its prose
- [ ] Co-authors have read and approved

---

## What is genuinely uncertain

**Leftover A changes the data-generating process for evaluation.** Every live
table was generated with no terminal admit (mean `|A|=767`). Setting
`sim.terminal_admit: true` admits arrivals in `(T-L, T]` as unserved. Absolute
J, SFR, and Table 7 will move. Rankings may or may not. Do not keep pre-admit
numbers next to post-admit numbers without saying so.

**PPO $449.60$ / $78\%$ gap-closed is test-scored.** Table 6 keeps untuned
`ppo_fair`. Leftover E produces the calib-selected row that the paper currently
flags as missing.

**Do not run the completeness evals on this OneDrive clone.** Campaign compute
is `C:\CAOR`.
