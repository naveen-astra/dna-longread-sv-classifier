"""Groups C and V -- caller-reported evidence and caller-declared type/length.

These are the ablated groups. A model that depends on them is closer to a
recalibration of the callers than to independent evidence, which is why the
`B+A` ablation exists (finding M3).
"""

from __future__ import annotations

import math
from typing import Any

from svml.io.records import CandidateRecord

SNIFFLES_KEYS = ("sniffles", "sniffles2")
CUTESV_KEYS = ("cutesv",)


def _match_caller(per_caller: dict[str, dict[str, Any]], keys: tuple[str, ...]) -> dict | None:
    for name, payload in per_caller.items():
        lowered = name.lower()
        if any(key in lowered for key in keys):
            return payload
    return None


def extract(cand: CandidateRecord) -> dict[str, Any]:
    """Compute all Group C and Group V features for one candidate."""
    sniffles = _match_caller(cand.per_caller, SNIFFLES_KEYS)
    cutesv = _match_caller(cand.per_caller, CUTESV_KEYS)

    out: dict[str, Any] = {}

    # -- Group C: caller-reported evidence --------------------------------------
    out["sniffles_support"] = _as_float(sniffles.get("support")) if sniffles else None
    out["cutesv_support"] = _as_float(cutesv.get("support")) if cutesv else None
    out["n_callers"] = len(cand.per_caller) if cand.per_caller else len(cand.callers)

    both = sniffles is not None and cutesv is not None
    if both:
        out["bp_offset_between_callers"] = abs(
            float(sniffles.get("pos", cand.pos)) - float(cutesv.get("pos", cand.pos))
        )
        out["svlen_disagreement"] = abs(
            abs(float(sniffles.get("svlen", cand.svlen)))
            - abs(float(cutesv.get("svlen", cand.svlen)))
        )
    else:
        # Structurally undefined with a single caller -- not zero. Zero would mean
        # "the two callers agreed perfectly", the opposite of what is true here.
        out["bp_offset_between_callers"] = None
        out["svlen_disagreement"] = None

    quals = [
        _as_float(payload.get("qual"))
        for payload in cand.per_caller.values()
        if payload.get("qual") is not None
    ]
    quals = [q for q in quals if q is not None]
    out["caller_qual"] = (sum(quals) / len(quals)) if quals else None

    out["filter_pass"] = int(
        cand.filter_pass or any(bool(p.get("filter_pass")) for p in cand.per_caller.values())
    )

    # -- Group V: caller-declared type and size ---------------------------------
    out["log10_svlen"] = (
        math.log10(abs(cand.svlen) + 1) if cand.svtype != "BND" or cand.svlen else None
    )
    out["svtype_called"] = cand.svtype
    out["span_len"] = cand.span_len

    return out


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
