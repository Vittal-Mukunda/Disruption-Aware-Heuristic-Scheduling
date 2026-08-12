"""Preflight: compile and import every module, before anything expensive runs.

Two seconds of work that catches the entire class of defect a five-hour campaign
would otherwise surface at hour three — syntax errors, bad imports, symbols that
moved, module-level code that raises.

It exists because the revision was written on a machine with no Python
interpreter, so every change to this repository was verified by reading. Reading
resolves imports and balances brackets; it does not catch a `TypeError` in a
default argument or a constant that was renamed in one file and not another.

    python scripts/preflight.py

Exit code 0 means every module in the package compiles and imports cleanly. It
does NOT mean the pipeline produces correct numbers — that is what the smoke
stages after it are for.
"""

from __future__ import annotations

import compileall
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("simulation", "labeling", "regime", "models", "baselines", "experiments")
TOP_LEVEL = ("seed", "logging_setup")


def _compile() -> list[str]:
    """Byte-compile everything; returns the list of files that failed."""
    failures: list[str] = []
    for pkg in (*PACKAGES, "tests"):
        d = REPO_ROOT / pkg
        if not d.exists():
            continue
        if not compileall.compile_dir(str(d), quiet=1, force=True):
            failures.append(pkg)
    for name in TOP_LEVEL:
        f = REPO_ROOT / f"{name}.py"
        if f.exists() and not compileall.compile_file(str(f), quiet=1, force=True):
            failures.append(f.name)
    return failures


def _import_all() -> list[tuple[str, str]]:
    """Import every module in the package tree; returns (module, error) pairs."""
    bad: list[tuple[str, str]] = []
    targets: list[str] = list(TOP_LEVEL)
    for pkg in PACKAGES:
        targets.append(pkg)
        try:
            mod = importlib.import_module(pkg)
        except Exception:
            bad.append((pkg, traceback.format_exc(limit=3).strip()))
            continue
        for m in pkgutil.iter_modules(mod.__path__, pkg + "."):
            targets.append(m.name)

    for name in targets:
        try:
            importlib.import_module(name)
        except Exception:
            bad.append((name, traceback.format_exc(limit=3).strip()))
    return bad


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    print(f"python  : {sys.version.split()[0]}")
    print(f"repo    : {REPO_ROOT}")

    for dep in ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "xgboost",
                "omegaconf", "joblib", "matplotlib"):
        try:
            m = importlib.import_module(dep)
            print(f"  {dep:<12} {getattr(m, '__version__', 'n/a')}")
        except Exception as exc:
            print(f"  {dep:<12} MISSING ({exc.__class__.__name__})")

    print("\n[1/2] byte-compiling...")
    comp = _compile()
    if comp:
        print(f"  SYNTAX ERRORS in: {comp}")
        return 1
    print("  ok — every file compiles")

    print("\n[2/2] importing every module...")
    bad = _import_all()
    if bad:
        print(f"  {len(bad)} MODULE(S) FAILED TO IMPORT:\n")
        for name, err in bad:
            print(f"--- {name} ---")
            print(err)
            print()
        return 1
    print("  ok — every module imports")

    print("\nPREFLIGHT PASSED. Safe to run the smoke stages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
