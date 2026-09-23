"""The confident-region gate (finding H4).

The single most important correctness rule in the project:

    A candidate that does not match the truth VCF is NOT automatically a false
    positive. It is a false positive only if it lies INSIDE the benchmark's confident
    regions. Outside them the benchmark makes no claim, so the candidate is UNKNOWN.

Labelling unknowns as negatives would teach the model that real SVs in hard regions are
artefacts -- exactly inverting the project's stated goal of performing well in
difficult and repetitive regions.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

from svml.io.bed import IntervalIndex
from svml.utils.logging import get_logger

log = get_logger(__name__)


class LabelStatus(str, Enum):
    INSIDE = "inside_confident"
    OUTSIDE = "outside_confident"


def apply_gate(
    frame: pd.DataFrame,
    confident_regions: IntervalIndex,
    require_full_span: bool = False,
) -> pd.DataFrame:
    """Add a `label_status` column marking each candidate inside or outside confident regions.

    `require_full_span=False` (the default) treats a candidate as inside when its
    breakpoint falls in a confident region. Requiring the entire span to be confident
    would discard most large events, since confident regions are fragmented.
    """
    if frame.empty:
        frame["label_status"] = pd.Series(dtype="object")
        return frame

    statuses: list[str] = []
    for row in frame.itertuples(index=False):
        chrom, pos, end = row.chrom, int(row.pos), int(row.end)
        if require_full_span and end > pos:
            inside = confident_regions.overlap_fraction(chrom, pos, end) >= 1.0
        else:
            inside = confident_regions.contains_point(chrom, pos)
        statuses.append(LabelStatus.INSIDE.value if inside else LabelStatus.OUTSIDE.value)

    frame = frame.copy()
    frame["label_status"] = statuses

    counts = frame["label_status"].value_counts().to_dict()
    total = len(frame)
    inside = counts.get(LabelStatus.INSIDE.value, 0)
    log.info(
        "confident-region gate: %d/%d inside (%.1f%%), %d outside (excluded from train and test)",
        inside,
        total,
        100 * inside / max(1, total),
        counts.get(LabelStatus.OUTSIDE.value, 0),
    )
    if inside == 0:
        log.error(
            "no candidates fall inside confident regions -- check contig naming between the "
            "candidate VCF and the benchmark BED, and confirm the pilot region is covered"
        )
    elif inside / max(1, total) < 0.10:
        log.warning(
            "only %.1f%% of candidates are inside confident regions; the labelled set may be "
            "too small to train on. Reconsider the pilot region (Phase 8 exit test).",
            100 * inside / max(1, total),
        )
    return frame


def gate_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Three-bucket counts for the report. This is a first-class limitation figure."""
    if "label_status" not in frame.columns:
        return {}
    inside_mask = frame["label_status"] == LabelStatus.INSIDE.value
    is_true = frame.get("is_true")

    counts = {
        "total_candidates": int(len(frame)),
        "inside_confident": int(inside_mask.sum()),
        "outside_confident_unlabelled": int((~inside_mask).sum()),
    }
    if is_true is not None:
        counts["labelled_true"] = int((is_true == 1).sum())
        counts["labelled_false"] = int((is_true == 0).sum())
    return counts
