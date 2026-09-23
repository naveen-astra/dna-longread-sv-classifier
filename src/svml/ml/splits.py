"""Chromosome-grouped train/validation/test splitting (finding H3).

A sample-level split is impossible here: only HG002 has SV truth. So the split groups
by chromosome, which guarantees that no genomic locus -- and therefore no cluster of
near-identical candidates from the same event -- straddles two splits.

`tests/test_splits.py` asserts the disjointness invariant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from svml.utils.logging import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")


class SplitError(ValueError):
    pass


def assign_splits(frame: pd.DataFrame, split_map: dict[str, str]) -> pd.DataFrame:
    """Add a `split` column from a chromosome -> split mapping.

    Candidates on a chromosome absent from the mapping get `split = NaN` and are
    excluded from modelling rather than silently landing in training.
    """
    frame = frame.copy()
    frame["split"] = frame["chrom"].map(split_map)

    unmapped = frame["split"].isna()
    if unmapped.any():
        chroms = sorted(frame.loc[unmapped, "chrom"].unique())
        log.warning(
            "%d candidates on unmapped contigs excluded from modelling: %s",
            int(unmapped.sum()),
            chroms[:10],
        )

    counts = frame["split"].value_counts(dropna=True).to_dict()
    log.info("split sizes: %s", counts)
    return frame


def validate_splits(frame: pd.DataFrame) -> None:
    """Assert the leakage invariant: no chromosome appears in more than one split."""
    labelled = frame[frame["split"].notna()]
    if labelled.empty:
        raise SplitError("no candidates were assigned to any split")

    by_chrom = labelled.groupby("chrom")["split"].nunique()
    straddling = by_chrom[by_chrom > 1]
    if not straddling.empty:
        raise SplitError(
            f"chromosome(s) {list(straddling.index)} appear in multiple splits. "
            "Chromosome-grouped splitting is the leakage control (finding H3)."
        )

    present = set(labelled["split"].unique())
    missing = set(SPLITS) - present
    if missing:
        raise SplitError(
            f"split(s) {sorted(missing)} are empty. Check regions.tier2 in config.yaml "
            f"against the chromosomes actually present in the data ({sorted(labelled['chrom'].unique())[:10]})."
        )


def modelling_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows usable for supervised learning on `target`.

    Drops rows outside confident regions (`is_true` is NA) and rows with no split.
    This is where the H4 gate actually takes effect on the training data.
    """
    subset = frame[frame["split"].notna()]
    subset = subset[subset[target].notna()]
    log.info(
        "modelling frame for target '%s': %d of %d rows usable",
        target,
        len(subset),
        len(frame),
    )
    return subset


def split_frames(
    frame: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = modelling_frame(frame, target)
    return (
        subset[subset["split"] == "train"],
        subset[subset["split"] == "val"],
        subset[subset["split"] == "test"],
    )


def class_distribution(frame: pd.DataFrame, target: str) -> dict[str, dict[str, int]]:
    """Per-split class counts. Reported before any imbalance remedy is chosen."""
    out: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        rows = frame[(frame["split"] == split) & frame[target].notna()]
        counts = rows[target].value_counts().to_dict()
        out[split] = {str(k): int(v) for k, v in counts.items()}
    return out


def write_splits(frame: pd.DataFrame, out_path: str | Path, split_map: dict[str, str]) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": "chromosome_grouped",
        "rationale": (
            "Sample-level split impossible: only HG002 has SV truth (finding H3). "
            "Grouping by chromosome prevents near-identical candidates from the same "
            "locus appearing in both training and test data."
        ),
        "split_map": split_map,
        "counts": {
            split: int((frame["split"] == split).sum()) for split in SPLITS
        },
        "chromosomes_present": sorted(frame["chrom"].dropna().unique().tolist()),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote split definition to %s", path)
    return path
