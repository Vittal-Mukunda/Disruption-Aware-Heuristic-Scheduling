"""Inline the exported model bundle into the single-file React artifact.

demo/export_model.py  ->  demo/dahs_model_bundle.json
demo/inline_model.py  ->  injects that JSON into demo/dahs_dashboard.jsx so the
                          .jsx is fully self-contained (no external imports).

The bundle is embedded as `JSON.parse("<escaped json string>")`. A single string
literal is one token for any JS parser (Babel/esbuild), whereas a 2.4 MB object
literal forces the parser to build hundreds of thousands of AST nodes -- which
hangs Babel-based toolchains. The runtime JSON.parse costs only tens of ms.

The artifact has a line `const MODEL = /*INJECT*/{};`. This script replaces
everything from the `/*INJECT*/` marker to the next `;`. Idempotent.

Run:  python demo/inline_model.py
"""

import json
import re
from pathlib import Path

demo = Path(__file__).resolve().parent
jsx_path = demo / "dahs_dashboard.jsx"
bundle_text = (demo / "dahs_model_bundle.json").read_text(encoding="utf-8")
jsx = jsx_path.read_text(encoding="utf-8")

if "/*INJECT*/" not in jsx:
    raise SystemExit("ERROR: /*INJECT*/ marker not found in dahs_dashboard.jsx")

# json.dumps turns the bundle text into a valid, fully-escaped JS string literal.
embed = "JSON.parse(" + json.dumps(bundle_text) + ")"

new = re.sub(
    r"/\*INJECT\*/.*?;",
    lambda _m: "/*INJECT*/" + embed + ";",
    jsx,
    count=1,
    flags=re.DOTALL,
)
jsx_path.write_text(new, encoding="utf-8")
print(f"Injected model bundle ({len(bundle_text):,} chars JSON -> {len(embed):,} chars embed).")
print(f"dahs_dashboard.jsx is now {len(new):,} chars ({len(new) / 1e6:.2f} MB).")
