"""BAM querying via pysam.

Provides the alignment primitives the Group B features are built from. Kept separate
from feature definitions so the read-level logic can be unit-tested against a tiny
fixture BAM independently of the feature schema.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from svml.utils.logging import get_logger

log = get_logger(__name__)


def _pysam():
    """Import pysam on demand, with an actionable message when it is unavailable."""
    try:
        import pysam
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pysam is required for BAM access but is not installed. It has no native "
            "Windows build -- run inside WSL2 or Docker with the svml conda environment "
            "activated (see docs/06_execution_plan.md, Phase 0)."
        ) from exc
    return pysam

# CIGAR operation codes (pysam convention)
CIGAR_MATCH = 0
CIGAR_INS = 1
CIGAR_DEL = 2
CIGAR_SOFT_CLIP = 4
CIGAR_HARD_CLIP = 5
CLIP_OPS = {CIGAR_SOFT_CLIP, CIGAR_HARD_CLIP}


@dataclass
class ReadSummary:
    """Per-read facts gathered in a single pass over a window."""

    mapq: int
    is_reverse: bool
    is_supplementary: bool
    has_sa_tag: bool
    left_clip: int
    right_clip: int
    aligned_length: int
    nm: int | None
    max_insertion: int
    max_deletion: int
    ref_start: int
    ref_end: int

    @property
    def max_clip(self) -> int:
        return max(self.left_clip, self.right_clip)

    @property
    def nm_rate(self) -> float | None:
        if self.nm is None or self.aligned_length <= 0:
            return None
        return self.nm / self.aligned_length


class BamReader:
    """Thin, reusable wrapper around an indexed BAM.

    Opened once per sample and reused across candidates -- reopening per candidate
    dominates runtime at Tier-2 scale.
    """

    def __init__(self, bam_path: str | Path, max_reads_per_locus: int = 1000) -> None:
        self.path = Path(bam_path)
        if not self.path.exists():
            raise FileNotFoundError(f"BAM not found: {self.path}")
        self._bam = _pysam().AlignmentFile(str(self.path), "rb")
        self.max_reads = max_reads_per_locus
        self._truncated_loci = 0

    def close(self) -> None:
        self._bam.close()

    def __enter__(self) -> BamReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def contigs(self) -> set[str]:
        return set(self._bam.references)

    @property
    def truncated_loci(self) -> int:
        """Loci where the read cap was hit. Logged so the cap's impact is visible."""
        return self._truncated_loci

    def _clamp(self, chrom: str, start: int, end: int) -> tuple[int, int] | None:
        if chrom not in self.contigs:
            return None
        length = self._bam.get_reference_length(chrom)
        start = max(0, start)
        end = min(length, end)
        if end <= start:
            return None
        return start, end

    def summarise_reads(self, chrom: str, start: int, end: int) -> list[ReadSummary]:
        """Summarise every primary-or-supplementary read overlapping [start, end)."""
        window = self._clamp(chrom, start, end)
        if window is None:
            return []
        start, end = window

        out: list[ReadSummary] = []
        for read in self._bam.fetch(chrom, start, end):
            if read.is_unmapped or read.is_secondary or read.is_duplicate:
                continue
            if len(out) >= self.max_reads:
                self._truncated_loci += 1
                break

            cigar = read.cigartuples or []
            left_clip = cigar[0][1] if cigar and cigar[0][0] in CLIP_OPS else 0
            right_clip = cigar[-1][1] if cigar and cigar[-1][0] in CLIP_OPS else 0
            max_ins = max((ln for op, ln in cigar if op == CIGAR_INS), default=0)
            max_del = max((ln for op, ln in cigar if op == CIGAR_DEL), default=0)

            try:
                nm = read.get_tag("NM")
            except KeyError:
                nm = None

            out.append(
                ReadSummary(
                    mapq=int(read.mapping_quality),
                    is_reverse=bool(read.is_reverse),
                    is_supplementary=bool(read.is_supplementary),
                    has_sa_tag=read.has_tag("SA"),
                    left_clip=int(left_clip),
                    right_clip=int(right_clip),
                    aligned_length=int(read.reference_length or 0),
                    nm=int(nm) if nm is not None else None,
                    max_insertion=int(max_ins),
                    max_deletion=int(max_del),
                    ref_start=int(read.reference_start),
                    ref_end=int(read.reference_end or read.reference_start),
                )
            )
        return out

    def mean_depth(self, chrom: str, start: int, end: int) -> float:
        """Mean per-base depth over [start, end).

        Counts all mapped reads including MAPQ 0 -- repetitive-region depth is signal
        here, not noise, and `frac_mapq0` captures the ambiguity separately.
        """
        window = self._clamp(chrom, start, end)
        if window is None:
            return 0.0
        start, end = window
        total = 0
        for column in self._bam.pileup(
            chrom, start, end, truncate=True, min_base_quality=0, max_depth=8000
        ):
            total += column.nsegments
        return total / (end - start)

    def depth_profile(self, chrom: str, start: int, end: int) -> np.ndarray:
        """Per-base depth array over [start, end). Empty array if out of range."""
        window = self._clamp(chrom, start, end)
        if window is None:
            return np.zeros(0, dtype=np.int32)
        start, end = window
        profile = np.zeros(end - start, dtype=np.int32)
        for column in self._bam.pileup(
            chrom, start, end, truncate=True, min_base_quality=0, max_depth=8000
        ):
            idx = column.reference_pos - start
            if 0 <= idx < profile.size:
                profile[idx] = column.nsegments
        return profile

    def spanning_reads(
        self, chrom: str, start: int, end: int, min_indel: int = 50
    ) -> tuple[int, int]:
        """Count reads crossing [start, end) with and without a large indel.

        Returns (n_spanning_clean, n_spanning_with_indel). A clean spanning read is
        reference-supporting evidence: it crossed the locus and saw no event.
        """
        window = self._clamp(chrom, max(0, start - 1), end + 1)
        if window is None:
            return 0, 0

        clean = with_indel = 0
        for read in self._bam.fetch(chrom, window[0], window[1]):
            if read.is_unmapped or read.is_secondary or read.is_duplicate:
                continue
            ref_end = read.reference_end or read.reference_start
            if read.reference_start > start or ref_end < end:
                continue
            cigar = read.cigartuples or []
            has_large = any(op in (CIGAR_INS, CIGAR_DEL) and ln >= min_indel for op, ln in cigar)
            if has_large:
                with_indel += 1
            else:
                clean += 1
        return clean, with_indel


def safe_mean(values: list[float] | list[int]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def safe_median(values: list[float] | list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def mad(values: np.ndarray) -> float:
    """Median absolute deviation. Returns 0.0 for an empty profile."""
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))
