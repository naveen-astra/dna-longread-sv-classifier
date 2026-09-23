"""The confident-region gate: the three-way label outcome (finding H4).

The rule under test: a candidate with no truth match is a false positive ONLY inside
confident regions. Outside them it is unknown, and labelling it 0 would teach the model
that real SVs in hard regions are artefacts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from svml.io.bed import IntervalIndex
from svml.labels.gate import LabelStatus, apply_gate, gate_counts


@pytest.fixture
def confident() -> IntervalIndex:
    return IntervalIndex({"chr1": [(1000, 2000), (5000, 6000)]}, name="confident")


@pytest.fixture
def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "pos": [1500, 3000, 5500, 1500],   # inside, outside, inside, other contig
            "end": [1600, 3100, 5600, 1600],
            "svtype": ["DEL"] * 4,
        }
    )


def test_gate_marks_inside_and_outside(candidates, confident):
    frame = apply_gate(candidates, confident)
    assert frame["label_status"].tolist() == [
        LabelStatus.INSIDE.value,
        LabelStatus.OUTSIDE.value,
        LabelStatus.INSIDE.value,
        LabelStatus.OUTSIDE.value,   # contig absent from the confident set
    ]


def test_outside_candidates_get_na_not_zero(candidates, confident):
    """The central correctness rule of the project."""
    frame = apply_gate(candidates, confident)

    truth_matches = {"a"}
    frame["is_true"] = [
        1.0 if row.candidate_id in truth_matches
        else (0.0 if row.label_status == LabelStatus.INSIDE.value else None)
        for row in frame.itertuples(index=False)
    ]

    assert frame.loc[frame["candidate_id"] == "a", "is_true"].iloc[0] == 1.0
    # Inside a confident region, unmatched -> genuine false positive.
    assert frame.loc[frame["candidate_id"] == "c", "is_true"].iloc[0] == 0.0
    # Outside -> unknown, NOT a false positive.
    assert frame.loc[frame["candidate_id"] == "b", "is_true"].isna().iloc[0]
    assert frame.loc[frame["candidate_id"] == "d", "is_true"].isna().iloc[0]


def test_gate_counts_reports_three_buckets(candidates, confident):
    frame = apply_gate(candidates, confident)
    frame["is_true"] = [1.0, None, 0.0, None]

    counts = gate_counts(frame)
    assert counts["total_candidates"] == 4
    assert counts["inside_confident"] == 2
    assert counts["outside_confident_unlabelled"] == 2
    assert counts["labelled_true"] == 1
    assert counts["labelled_false"] == 1


def test_require_full_span_is_stricter(confident):
    """A candidate spanning outside a confident region fails the strict gate."""
    frame = pd.DataFrame(
        {
            "candidate_id": ["big"],
            "chrom": ["chr1"],
            "pos": [1500],
            "end": [2500],     # runs past the confident interval ending at 2000
            "svtype": ["DEL"],
        }
    )
    lenient = apply_gate(frame, confident, require_full_span=False)
    strict = apply_gate(frame, confident, require_full_span=True)

    assert lenient["label_status"].iloc[0] == LabelStatus.INSIDE.value
    assert strict["label_status"].iloc[0] == LabelStatus.OUTSIDE.value


def test_empty_frame_is_handled():
    frame = pd.DataFrame(columns=["candidate_id", "chrom", "pos", "end", "svtype"])
    result = apply_gate(frame, IntervalIndex({}, name="empty"))
    assert "label_status" in result.columns
    assert result.empty
