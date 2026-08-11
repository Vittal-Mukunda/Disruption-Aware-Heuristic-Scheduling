"""Label provenance stamps.

A label parquet is not self-describing. `data/train.parquet` from the submitted
campaign and one produced by the current Stage 2 have the same column shape —
`shift_id`, the `f_*` features, a `p_*` column per rule — so every schema check
passes on both, and `models.heuristic_ranker.pool_from_frame` reads a perfectly
clean pool out of either. Nothing in the file records which objective, which
rule pool, or which rollout estimator produced it.

That matters because the revision changed all three. Training on a stale parquet
does not fail; it succeeds and yields a model fitted to labels generated under
the cost function this revision exists to replace.

So Stage 2 writes a stamp beside the parquet and Stage 3 refuses to run without
one. Any driver that *derives* a parquet from a stamped one — a shift subset for
the data-efficiency sweep, one-hot labels for the `hard_labels` ablation — must
carry the stamp forward with `stamp_derived`, which also records what it did, so
the chain from a trained model back to the labelling run stays readable.
"""

from __future__ import annotations

import json
from pathlib import Path

LABEL_META_NAME: str = "label_meta.json"


def meta_path_for(parquet_path: Path | str) -> Path:
    """The stamp that belongs to `parquet_path` (a sibling file)."""
    return Path(parquet_path).parent / LABEL_META_NAME


def write_label_meta(parquet_path: Path | str, meta: dict) -> Path:
    """Write the stamp beside a freshly produced label parquet."""
    out = meta_path_for(parquet_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    return out


def read_label_meta(parquet_path: Path | str) -> dict:
    """Load the stamp, or explain what is missing and how to produce it."""
    p = Path(parquet_path)
    meta = meta_path_for(p)
    if not meta.exists():
        raise SystemExit(
            f"No {LABEL_META_NAME} beside {p}.\n"
            f"These labels were not produced by the current Stage 2, so they "
            f"predate the corrected objective, the two-clock order model and the "
            f"screened rule pool. Their columns are still well formed, so "
            f"training on them would SUCCEED and produce a wrong model.\n"
            f"  - for the main corpus: `python -m experiments.generate_labels` "
            f"(and `make clean-stale` first)\n"
            f"  - for a derived corpus: the driver that wrote {p.name} should "
            f"call labeling.provenance.stamp_derived()"
        )
    return json.loads(meta.read_text(encoding="utf-8"))


def stamp_derived(
    src_parquet: Path | str,
    dst_parquet: Path | str,
    derivation: str,
    **extra,
) -> Path:
    """Carry a stamp from `src_parquet` to a parquet derived from it.

    `derivation` is a short human-readable description of the transform — it
    appears in the stamp so a run directory can be traced back through every
    intermediate to the labelling pass that produced its inputs.
    """
    meta = dict(read_label_meta(src_parquet))
    meta["derived_from"] = str(src_parquet)
    meta["derivation"] = [*meta.get("derivation", []), derivation]
    meta.update(extra)
    return write_label_meta(dst_parquet, meta)
