"""Package the manuscript for submission to Computers & Operations Research.

Two jobs, and the first is the one that catches mistakes:

    python scripts/build_submission.py --check    # readiness gate, no output files
    python scripts/build_submission.py            # strip scaffolding + convert

--check is safe to run at any time and is worth running BEFORE the campaign, so
the failures you are going to hit are known early rather than at submission.

WHAT IT REFUSES TO BUILD
------------------------
A submission with unresolved TBD-rerun markers, or with draft scaffolding still
in it, is not a submission. The gate blocks on both. It also blocks on superseded
numbers appearing outside a table explicitly labelled "(superseded)", because that
is the specific failure mode this revision exists to correct: the submitted paper
reported a metric whose denominator excluded the orders the controller declined to
serve, and carrying any of those figures forward as a live claim would repeat it.

TOOLCHAIN
---------
Conversion needs pandoc and a LaTeX engine, neither of which is installed on the
development laptop. The gate reports their absence as a WARNING rather than an
error so it stays useful without them; only the conversion itself hard-fails.

    pandoc      https://pandoc.org/installing.html
    MiKTeX      https://miktex.org  (Windows)
    elsarticle  ships with TeX Live / MiKTeX; `tlmgr install elsarticle` otherwise
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# The console on Windows defaults to cp1252 and the TBD markers are U+27E8/9,
# so printing a problem list would crash before showing it. Reconfigure first.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
MS = PAPER / "manuscript.md"
BIB = PAPER / "references.bib"
OUT = PAPER / "submission"

# Numbers from the submitted version. Any of these appearing OUTSIDE a table
# marked "(superseded)" is a live claim on the retracted objective.
SUPERSEDED = [
    r"1\.33\s*%", r"0\.0133", r"1\.44\s*%", r"\b3\.09\b", r"\b7\.18\b",
    r"0\.0373", r"2\.40\s+points", r"\b5\.85\b", r"0\.063 to 0\.028",
]

LANG, RANG = chr(0x27E8), chr(0x27E9)

SCAFFOLDING = [
    ("revision-note blockquote", r"> \*\*Revision note on pending numbers\.\*\*"),
    ("DRAFT vN comment block", r"<!--\s*\nDRAFT v"),
]


def _strip_superseded_tables(text: str) -> str:
    """Remove every markdown table that a '(superseded)' caption introduces.

    The caption line names the table; the block that follows, up to the next
    blank line after the table body, is the table itself. Removing both lets the
    live-claim scan run over prose only.
    """
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        if "(superseded)" in lines[i].lower():
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith("|"):
                i += 1                      # caption continuation
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1                      # the table
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def check() -> list[str]:
    """Return a list of blocking problems. Empty list means ready."""
    problems: list[str] = []
    if not MS.exists():
        return [f"{MS} not found"]
    text = MS.read_text(encoding="utf-8")

    n_tbd = text.count("TBD-rerun")
    if n_tbd:
        problems.append(
            f"{n_tbd} unresolved TBD-rerun marker(s). Each states what to report "
            f"and which way the conclusion falls; resolve against the measured "
            f"outcome rather than deleting the unfavourable branch."
        )

    for name, pattern in SCAFFOLDING:
        if re.search(pattern, text):
            problems.append(f"draft scaffolding still present: {name}")

    prose = _strip_superseded_tables(text)
    prose = re.sub(r"<!--.*?-->", "", prose, flags=re.S)
    for pattern in SUPERSEDED:
        for m in re.finditer(pattern, prose):
            # Look at the whole paragraph, not the line: a number attributed to
            # the submitted version ("the submitted Table 1 read X") is exactly
            # how those figures are supposed to be discussed. Only an
            # unattributed one is a live claim.
            para_start = prose.rfind("\n\n", 0, m.start()) + 2
            para_end = prose.find("\n\n", m.start())
            para = prose[para_start:para_end if para_end > 0 else len(prose)]
            if re.search(r"submitted|superseded|previous version|old objective",
                         para, re.I):
                continue
            s0 = prose.rfind("\n", 0, m.start()) + 1
            line = prose[s0:prose.find("\n", m.start())].strip()
            problems.append(
                f"superseded number as a live claim: {line[:80]!r}"
            )

    # Editorial scaffolding may mention reviewers; the ARTICLE may not. Strip
    # comments, blockquote notes and TBD spans first, exactly as
    # scripts/audit_reviewer_items.py does, so the two agree.
    art = re.sub(r"<!--.*?-->", "", prose, flags=re.S)
    art = re.sub(LANG + r"TBD-rerun.*?" + RANG, "", art, flags=re.S)
    art = chr(10).join(
        ln for ln in art.splitlines() if not ln.lstrip().startswith(">")
    )
    for m in re.finditer(r"[Rr]eviewer", art):
        s0 = art.rfind(chr(10), 0, m.start()) + 1
        problems.append(
            "the article addresses a reviewer in its prose (R1.7d): "
            + art[s0:art.find(chr(10), m.start())].strip()[:60]
        )

    # Elsevier: tables and figures numbered consecutively in order of first
    # mention. The revision broke both — Sections 3.5 and 6.1 gained tables that
    # appear before the old Table 1, and figures deleted during the revision left
    # gaps. It will break again as the campaign adds and removes results, so this
    # is checked rather than fixed once.
    #
    # References to the SUBMITTED paper's tables are a different document's
    # numbering and must NOT be renumbered; they are matched and excluded here so
    # a legitimate "the submitted Table 1" does not trip the ordering check.
    article = re.sub(r"the submitted (?:paper's )?Table\s+\d+", "", prose, flags=re.I)
    for kind in ("Table", "Figure"):
        seen = []
        for m in re.finditer(rf"{kind}\s+(\d+)", article):
            if m.group(1) not in seen:
                seen.append(m.group(1))
        if not seen:
            continue
        expected = [str(i) for i in range(1, len(seen) + 1)]
        if seen != expected:
            nums = sorted(int(n) for n in seen)
            gaps = [n for n in range(1, max(nums) + 1) if n not in nums]
            detail = f"gaps at {gaps}" if gaps else "out of order of first mention"
            problems.append(
                f"{kind} numbering is not consecutive in order of first mention "
                f"({detail}); saw {seen}"
            )
        # Every number mentioned must also have a caption.
        if kind == "Table":
            caps = set(re.findall(r"^\*\*Table (\d+)", text, re.M))
        else:
            caps = set(re.findall(r"^!\[Figure (\d+)", text, re.M))
        uncaptioned = sorted(set(seen) - caps, key=int)
        if uncaptioned:
            problems.append(
                f"{kind}(s) referenced without a caption: {uncaptioned}"
            )

    for fig in sorted(set(re.findall(r"\]\((\.\./figures/[^)]+)\)", text))):
        if not (PAPER / fig).resolve().exists():
            problems.append(f"figure referenced but missing: {fig}")

    # Bibliography closure, both directions.
    if BIB.exists():
        keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
        body = text.split("\n---", 2)[-1] if text.startswith("---") else text
        used = {k.rstrip(".,;:") for k in
                re.findall(r"@([A-Za-z][A-Za-z0-9_:.+-]*)", body)}
        if used - keys:
            problems.append(f"cited but not in the bibliography: {sorted(used - keys)}")
        if keys - used:
            problems.append(f"in the bibliography but never cited: {sorted(keys - used)}")

    for name, path in (
        ("response to reviewers", PAPER / "RESPONSE_TO_REVIEWERS.md"),
        ("cover letter", PAPER / "COVER_LETTER.md"),
    ):
        if not path.exists():
            problems.append(f"{name} missing: {path}")
        elif "⟨" in path.read_text(encoding="utf-8"):
            problems.append(f"{name} still has unfilled ⟨…⟩ slots")

    return problems


def toolchain() -> list[str]:
    """Non-blocking warnings about the conversion toolchain."""
    warns = []
    if shutil.which("pandoc") is None:
        warns.append("pandoc not on PATH — conversion unavailable")
    if not any(shutil.which(e) for e in ("pdflatex", "xelatex", "lualatex", "tectonic")):
        warns.append("no LaTeX engine on PATH — PDF build unavailable")
    return warns


def build() -> int:
    if shutil.which("pandoc") is None:
        print("[build] pandoc not found. Install it, then re-run.", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    text = MS.read_text(encoding="utf-8")

    # Strip scaffolding that must not reach a referee.
    text = re.sub(r"<!--\s*\nDRAFT v.*?-->", "", text, flags=re.S)
    text = re.sub(
        r"> \*\*Revision note on pending numbers\.\*\*.*?(?=\n\n# )",
        "", text, flags=re.S,
    )
    staged = OUT / "manuscript_clean.md"
    staged.write_text(text, encoding="utf-8")

    tex = OUT / "manuscript.tex"
    cmd = [
        "pandoc", str(staged),
        "--from", "markdown+tex_math_dollars+pipe_tables+raw_tex",
        "--to", "latex",
        "--standalone",
        "--template", str(PAPER / "elsarticle.template.tex"),
        "--citeproc", f"--bibliography={BIB}",
        "--number-sections",
        "-o", str(tex),
    ]
    print("[build]", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc:
        return rc
    print(f"[build] wrote {tex.relative_to(REPO)}")
    print("[build] now run your LaTeX engine on it; elsarticle must be installed.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Report readiness and exit without writing anything.")
    args = ap.parse_args()

    problems, warns = check(), toolchain()

    for w in warns:
        print(f"[warn ] {w}")
    if problems:
        print(f"\n[gate ] NOT READY — {len(problems)} blocking problem(s):")
        seen = set()
        for p in problems:
            if p not in seen:
                print(f"   - {p}")
                seen.add(p)
        print("\nSee SUBMISSION_CHECKLIST.md for the order to work through these.")
        return 1

    print("\n[gate ] READY — no blocking problems found.")
    if args.check:
        return 0
    return build()


if __name__ == "__main__":
    sys.exit(main())
