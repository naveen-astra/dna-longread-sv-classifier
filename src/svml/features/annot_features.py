"""Group A -- annotation-derived features (reference genomic context).

Supplies the `repeat_context` family the project spec names as a required feature
(spec section 6, objective O2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from svml.features.registry import SENTINEL_DISTANCE
from svml.io.bed import IntervalIndex
from svml.io.records import CandidateRecord
from svml.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - pysam has no native Windows build
    import pysam

log = get_logger(__name__)


@dataclass
class AnnotationSet:
    """The annotation tracks plus the indexed reference FASTA."""

    simple_repeat: IntervalIndex
    repeatmasker: IntervalIndex
    segdup: IntervalIndex
    gaps: IntervalIndex
    fasta: pysam.FastaFile | None = None

    def close(self) -> None:
        if self.fasta is not None:
            self.fasta.close()


def classify_chrom(chrom: str) -> str:
    name = chrom.replace("chr", "")
    if name == "X":
        return "chrX"
    if name == "Y":
        return "chrY"
    if name.isdigit() and 1 <= int(name) <= 22:
        return "autosome"
    return "other"


def extract(
    cand: CandidateRecord,
    annotations: AnnotationSet,
    w_flank: int = 1000,
) -> dict[str, Any]:
    """Compute all Group A features for one candidate."""
    chrom, pos = cand.chrom, cand.pos

    # For insertions the reference span is zero, so overlap fractions are measured over
    # a small window around the breakpoint instead -- an insertion still sits in a
    # repeat context even though it consumes no reference.
    if cand.has_reference_span:
        span_start, span_end = cand.pos, cand.end
    else:
        span_start, span_end = max(0, pos - 50), pos + 50

    out: dict[str, Any] = {
        "tr_overlap_frac": annotations.simple_repeat.overlap_fraction(chrom, span_start, span_end),
        "rmsk_overlap_frac": annotations.repeatmasker.overlap_fraction(chrom, span_start, span_end),
        "segdup_overlap": annotations.segdup.overlap_fraction(chrom, span_start, span_end),
        "dist_to_nearest_gap": annotations.gaps.distance_to_nearest(
            chrom, pos, sentinel=SENTINEL_DISTANCE
        ),
        "chrom_class": classify_chrom(chrom),
    }

    seq_features = _sequence_features(cand, annotations.fasta, w_flank, span_start, span_end)
    out.update(seq_features)
    return out


def _sequence_features(
    cand: CandidateRecord,
    fasta: pysam.FastaFile | None,
    w_flank: int,
    span_start: int,
    span_end: int,
) -> dict[str, Any]:
    if fasta is None:
        return {
            "gc_content": None,
            "homopolymer_len_at_bp": None,
            "bp_microhomology_len": None,
        }

    chrom = cand.chrom
    try:
        contig_len = fasta.get_reference_length(chrom)
    except KeyError:
        log.warning("contig %s absent from reference FASTA; sequence features unavailable", chrom)
        return {
            "gc_content": None,
            "homopolymer_len_at_bp": None,
            "bp_microhomology_len": None,
        }

    def fetch(start: int, end: int) -> str:
        start = max(0, start)
        end = min(contig_len, end)
        if end <= start:
            return ""
        return fasta.fetch(chrom, start, end).upper()

    # GC over the span plus flanks. Capped so a multi-megabase event does not force a
    # huge sequence fetch; the local composition is what matters.
    gc_start = max(0, span_start - w_flank)
    gc_end = min(span_end + w_flank, gc_start + 200_000)
    gc_seq = fetch(gc_start, gc_end)
    gc_content = _gc_fraction(gc_seq)

    bp_seq = fetch(cand.pos - 50, cand.pos + 50)
    homopolymer = _longest_homopolymer(bp_seq) if bp_seq else None

    left_flank = fetch(cand.pos - 50, cand.pos)
    right_flank = fetch(span_end, span_end + 50)
    microhomology = _microhomology_length(left_flank, right_flank)

    return {
        "gc_content": gc_content,
        "homopolymer_len_at_bp": homopolymer,
        "bp_microhomology_len": microhomology,
    }


def _gc_fraction(seq: str) -> float | None:
    if not seq:
        return None
    acgt = sum(1 for b in seq if b in "ACGT")
    if acgt == 0:
        return None
    gc = sum(1 for b in seq if b in "GC")
    return gc / acgt


def _longest_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    best = run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1] and seq[i] in "ACGT":
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _microhomology_length(left: str, right: str, max_len: int = 50) -> int:
    """Longest suffix of `left` that is also a prefix of `right`.

    Microhomology at a junction is a signature of the DNA repair mechanism that created
    a real SV; its absence at an apparent breakpoint suggests alignment ambiguity.
    """
    if not left or not right:
        return 0
    limit = min(len(left), len(right), max_len)
    for length in range(limit, 0, -1):
        if left[-length:] == right[:length]:
            return length
    return 0
