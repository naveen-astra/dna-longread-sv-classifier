"""The leakage invariant: no chromosome may span two splits (finding H3)."""

from __future__ import annotations

import pandas as pd
import pytest

from svml.ml.splits import (
    SplitError,
    assign_splits,
    class_distribution,
    modelling_frame,
    split_frames,
    validate_splits,
)


def test_splits_are_chromosome_disjoint(feature_frame, split_map):
    frame = assign_splits(feature_frame, split_map)
    validate_splits(frame)

    by_chrom = frame.groupby("chrom")["split"].nunique()
    assert (by_chrom == 1).all(), "a chromosome appeared in more than one split"


def test_no_candidate_id_spans_splits(feature_frame, split_map):
    frame = assign_splits(feature_frame, split_map)
    per_id = frame.groupby("candidate_id")["split"].nunique()
    assert (per_id <= 1).all()


def test_validate_rejects_straddling_chromosome(feature_frame):
    frame = assign_splits(feature_frame, {"chr1": "train", "chr2": "val", "chr3": "test"})
    # Force a violation.
    frame.loc[frame["chrom"] == "chr1", "split"] = ["train", "test"] * (
        int((frame["chrom"] == "chr1").sum() / 2)
    ) + ["train"] * ((frame["chrom"] == "chr1").sum() % 2)
    with pytest.raises(SplitError, match="multiple splits"):
        validate_splits(frame)


def test_validate_rejects_empty_split(feature_frame):
    frame = assign_splits(feature_frame, {"chr1": "train", "chr2": "val"})
    with pytest.raises(SplitError, match="empty"):
        validate_splits(frame)


def test_unlabelled_rows_are_excluded_from_modelling(feature_frame, split_map):
    """Rows outside confident regions carry is_true = NA and must not be trained on."""
    frame = assign_splits(feature_frame, split_map)
    frame.loc[frame.index[:100], "is_true"] = pd.NA

    subset = modelling_frame(frame, "is_true")
    assert len(subset) == len(frame) - 100
    assert subset["is_true"].notna().all()


def test_split_frames_partition_cleanly(feature_frame, split_map):
    frame = assign_splits(feature_frame, split_map)
    train, val, test = split_frames(frame, "is_true")

    assert not set(train["chrom"]) & set(val["chrom"])
    assert not set(train["chrom"]) & set(test["chrom"])
    assert not set(val["chrom"]) & set(test["chrom"])
    assert len(train) + len(val) + len(test) == frame["is_true"].notna().sum()


def test_class_distribution_reports_all_splits(feature_frame, split_map):
    frame = assign_splits(feature_frame, split_map)
    distribution = class_distribution(frame, "is_true")
    assert set(distribution) == {"train", "val", "test"}
