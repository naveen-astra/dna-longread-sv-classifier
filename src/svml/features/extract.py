"""Feature matrix construction.

One row per merged candidate SV. This module is the layer boundary the architecture
depends on: it reads the BAM, the reference, the annotations and the candidate VCF,
and it never reads truth data. Truth enters only at `svml.labels`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from svml.features import annot_features, bam_features, caller_features
from svml.features.annot_features import AnnotationSet
from svml.features.registry import (
    FEATURES,
    ID_COLUMNS,
    SENTINEL_DISTANCE,
    Missing,
)
from svml.io.bam import BamReader
from svml.io.bed import IntervalIndex
from svml.io.records import CandidateRecord
from svml.utils.logging import get_logger

log = get_logger(__name__)


def load_annotations(
    annotation_paths: dict[str, str | Path],
    reference_fasta: str | Path | None = None,
) -> AnnotationSet:
    """Load the Group A annotation tracks and the indexed reference."""
    fasta = None
    if reference_fasta is not None:
        import pysam

        ref_path = Path(reference_fasta)
        if ref_path.exists():
            fasta = pysam.FastaFile(str(ref_path))
        else:
            log.warning("reference FASTA %s not found; sequence features will be null", ref_path)

    def load(key: str) -> IntervalIndex:
        path = annotation_paths.get(key)
        if path is None or not Path(path).exists():
            log.warning("annotation track '%s' unavailable; its features will be zero", key)
            return IntervalIndex({}, name=key)
        return IntervalIndex.from_bed(path, name=key)

    return AnnotationSet(
        simple_repeat=load("simple_repeat"),
        repeatmasker=load("repeatmasker"),
        segdup=load("segmental_duplications"),
        gaps=load("assembly_gaps"),
        fasta=fasta,
    )


def build_feature_matrix(
    candidates: list[CandidateRecord],
    bam_path: str | Path,
    annotations: AnnotationSet,
    sample_id: str,
    w_bp: int = 200,
    w_flank: int = 1000,
    min_clip_length: int = 50,
    min_sv_length: int = 50,
    max_reads_per_locus: int = 1000,
) -> pd.DataFrame:
    """Extract every feature group for every candidate into a single DataFrame."""
    rows: list[dict[str, Any]] = []
    failures = 0

    with BamReader(bam_path, max_reads_per_locus=max_reads_per_locus) as bam:
        missing_contigs = {c.chrom for c in candidates} - bam.contigs
        if missing_contigs:
            log.warning(
                "%d candidate contig(s) absent from the BAM: %s -- check contig naming",
                len(missing_contigs),
                sorted(missing_contigs)[:5],
            )

        for i, cand in enumerate(candidates):
            if i and i % 1000 == 0:
                log.info("extracted features for %d/%d candidates", i, len(candidates))

            row: dict[str, Any] = {
                "candidate_id": cand.candidate_id,
                "sample_id": sample_id,
                "chrom": cand.chrom,
                "pos": cand.pos,
                "end": cand.end,
                "svtype": cand.svtype,
                "svlen": cand.svlen,
            }
            try:
                row.update(
                    bam_features.extract(
                        cand,
                        bam,
                        w_bp=w_bp,
                        w_flank=w_flank,
                        min_clip_length=min_clip_length,
                        min_sv_length=min_sv_length,
                    )
                )
                row.update(annot_features.extract(cand, annotations, w_flank=w_flank))
                row.update(caller_features.extract(cand))
            except Exception as exc:  # noqa: BLE001 - one bad locus must not kill the run
                failures += 1
                log.warning("feature extraction failed for %s: %s", cand.candidate_id, exc)
                for feat in FEATURES:
                    row.setdefault(feat.name, None)
            rows.append(row)

        if bam.truncated_loci:
            log.warning(
                "read cap (%d) hit at %d loci; depth-derived features there are truncated",
                max_reads_per_locus,
                bam.truncated_loci,
            )

    if failures:
        pct = 100 * failures / max(1, len(candidates))
        log.warning("feature extraction failed at %d loci (%.2f%%)", failures, pct)
        if pct > 1.0:
            log.error(
                "failure rate above 1%% indicates a pipeline bug, not missing data "
                "(docs/03_feature_dictionary.md, missing-value case 3)"
            )

    frame = pd.DataFrame(rows)
    frame = _finalise(frame)
    log.info("built feature matrix: %d rows x %d columns", len(frame), frame.shape[1])
    return frame


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the missing-value policy and guarantee every registered column exists."""
    if frame.empty:
        return pd.DataFrame(columns=ID_COLUMNS + [f.name for f in FEATURES])

    for feat in FEATURES:
        if feat.name not in frame.columns:
            frame[feat.name] = None

        if feat.missing is Missing.ZERO and not feat.categorical:
            # Genuinely zero: a real measurement, not missing data.
            frame[feat.name] = pd.to_numeric(frame[feat.name], errors="coerce").fillna(0)
        elif feat.missing is Missing.SENTINEL:
            frame[feat.name] = pd.to_numeric(frame[feat.name], errors="coerce").fillna(
                SENTINEL_DISTANCE
            )
        elif feat.categorical:
            frame[feat.name] = frame[feat.name].astype("object").where(frame[feat.name].notna(), "")
        # NAN_INDICATOR features keep their NaN here; the indicator column is added in
        # svml.ml.dataset, after the train/val/test split, so imputation statistics are
        # fit on training data only.

    ordered = ID_COLUMNS + [f.name for f in FEATURES]
    return frame[[c for c in ordered if c in frame.columns]]


def write_matrix(frame: pd.DataFrame, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log.info("wrote feature matrix to %s", path)
    return path
