"""Shared fixtures.

Fixtures are tiny and synthetic so the test suite runs without the bioinformatics
toolchain or any downloaded data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from svml.features.registry import FEATURES


@pytest.fixture
def split_map() -> dict[str, str]:
    return {
        "chr1": "train", "chr2": "train", "chr3": "train",
        "chr17": "val", "chr18": "val",
        "chr20": "test", "chr21": "test",
    }


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """A synthetic feature matrix with every registered column present."""
    rng = np.random.default_rng(0)
    n = 400
    chroms = ["chr1", "chr2", "chr3", "chr17", "chr18", "chr20", "chr21"]

    frame = pd.DataFrame(
        {
            "candidate_id": [f"SV_{i:05d}" for i in range(n)],
            "sample_id": "HG002_ONT",
            "chrom": rng.choice(chroms, n),
            "pos": rng.integers(1_000, 1_000_000, n),
            "svtype": rng.choice(["DEL", "INS"], n),
            "svlen": rng.integers(50, 5_000, n),
        }
    )
    frame["end"] = frame["pos"] + frame["svlen"]

    for feat in FEATURES:
        if feat.name in frame.columns:
            continue
        if feat.categorical:
            frame[feat.name] = "autosome" if feat.name == "chrom_class" else frame["svtype"]
        elif feat.dtype == "int":
            frame[feat.name] = rng.integers(0, 40, n)
        else:
            frame[feat.name] = rng.random(n)

    # A signal the model can actually learn, so training tests are not fitting noise.
    frame["is_true"] = (frame["dp_flank_ratio"] < 0.5).astype(float)
    frame["true_sv_type"] = np.where(frame["is_true"] == 1, frame["svtype"], None)
    frame["label_status"] = "inside_confident"
    frame["n_callers"] = rng.choice([1, 2], n)
    return frame


@pytest.fixture
def tiny_bed(tmp_path):
    path = tmp_path / "regions.bed"
    path.write_text(
        "chr1\t100\t200\n"
        "chr1\t150\t300\n"      # overlaps the previous interval; must be merged
        "chr1\t1000\t1100\n"
        "chr2\t500\t600\n",
        encoding="utf-8",
    )
    return path
