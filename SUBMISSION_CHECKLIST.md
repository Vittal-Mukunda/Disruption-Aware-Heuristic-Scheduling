# Manuscript checklist

Source of truth: `paper/manuscript.md`. Do not quote `CAMPAIGN_REPORT.md` or
any deleted runbook; those files are gone because they carried pre-admit
numbers.

1. `python scripts/audit_reviewer_items.py` — 40/40 phrase anchors.
2. `python scripts/build_submission.py --check` — READY (abstract ≤250 words,
   no TBD, no live superseded numbers, table/figure order, bib closure).
3. `python scripts/build_submission.py` — writes `paper/submission/manuscript.tex`
   (needs pandoc). Compile it with two `pdflatex` passes; pandoc inlines the
   bibliography with `--citeproc`, so there is no bibtex/biber pass.
4. `paper/manuscript.tex` and `paper/manuscript.pdf` are copies of
   `paper/submission/manuscript.{tex,pdf}` from that build. They are build
   output — regenerate, never hand-edit.
5. Highlights in `paper/highlights.txt` and `paper/elsarticle.template.tex` must
   match, five bullets, each ≤85 characters.
6. Live Table 5: DAHS $J=382.27$, FIFO $1486.82$ / SFR $0.1837$ / $3.89\times$,
   teachers $356.98$ / $363.42$, $|A|=791$, latency $4.24$ ms vs $670$ ms
   ($158\times$; $0.07\%$ of a 15-minute epoch).
7. Do not re-run leftover A–G or relabel. Do not run `scripts/clean_stale.py`.
