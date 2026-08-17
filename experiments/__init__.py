"""Experiment drivers.

STDOUT IS FORCED TO UTF-8 HERE, AND IT IS NOT COSMETIC. Twelve modules in this
repository print non-ASCII — arrows, `theta`, `tau`, `sigma`, em dashes — in
progress lines and result tables. On Windows the console and redirected stdout
default to cp1252 with `errors='strict'`, so `print` raises UnicodeEncodeError
and kills the driver. `experiments.e4_sensitivity theta` died exactly that way on
its own banner, having done no work and produced no output.

The campaign runs on Windows for ~16 hours, so a driver that crashes on a
progress line loses whatever preceded it. Reconfiguring the package's streams
once, at import, covers every `python -m experiments.X` entry point; the
standalone scripts under `scripts/` do the same thing for themselves.

`errors="replace"` rather than strict: a mangled glyph in a log line is a
cosmetic problem, and a dead driver at hour nine is not.
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Already detached, or not a real stream (pytest capture, some IDEs).
            pass
