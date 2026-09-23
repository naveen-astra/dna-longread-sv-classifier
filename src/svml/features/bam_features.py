"""Group B -- BAM-derived features.

Computed from alignments alone, independently of what the callers reported. This is
what makes the `B+A` ablation a meaningful test of whether independent evidence adds
anything beyond the callers' own confidence (finding M3).
"""

from __future__ import annotations

import math
from typing import Any

from svml.io.bam import BamReader, mad, safe_mean, safe_median
from svml.io.records import CandidateRecord
from svml.utils.logging import get_logger

log = get_logger(__name__)


def extract(
    cand: CandidateRecord,
    bam: BamReader,
    w_bp: int = 200,
    w_flank: int = 1000,
    min_clip_length: int = 50,
    min_sv_length: int = 50,
) -> dict[str, Any]:
    """Compute all Group B features for one candidate."""
    chrom, pos, end = cand.chrom, cand.pos, cand.end

    bp_start, bp_end = pos - w_bp, pos + w_bp
    reads = bam.summarise_reads(chrom, bp_start, bp_end)

    out: dict[str, Any] = {}

    # -- depth ------------------------------------------------------------------
    depth_profile = bam.depth_profile(chrom, bp_start, bp_end)
    out["dp_local"] = float(depth_profile.mean()) if depth_profile.size else 0.0
    out["dp_mad_local"] = mad(depth_profile)
    out["dp_flank_ratio"] = _depth_ratio(cand, bam, w_flank)

    # -- mapping quality --------------------------------------------------------
    mapqs = [r.mapq for r in reads]
    out["mean_mapq"] = safe_mean(mapqs)
    out["median_mapq"] = safe_median(mapqs)
    out["frac_mapq0"] = (sum(1 for m in mapqs if m == 0) / len(mapqs)) if mapqs else None

    # -- breakpoint evidence ----------------------------------------------------
    out["n_split_reads"] = sum(1 for r in reads if r.has_sa_tag)
    n_clipped = sum(1 for r in reads if r.max_clip >= min_clip_length)
    out["n_clipped_reads"] = n_clipped
    out["n_supplementary"] = sum(1 for r in reads if r.is_supplementary)

    dp = out["dp_local"]
    out["clip_frac"] = min(1.0, n_clipped / dp) if dp > 0 else None

    # -- spanning / support -----------------------------------------------------
    span_target_end = end if cand.has_reference_span else pos + max(1, min_sv_length)
    clean_spanning, spanning_with_indel = bam.spanning_reads(
        chrom, pos, span_target_end, min_indel=min_sv_length
    )
    out["n_spanning_reads"] = clean_spanning

    # Supporting reads: those showing the event, i.e. a large indel in the CIGAR or a
    # clip/split at the breakpoint. Deliberately computed from alignments, not from
    # the caller's own SUPPORT field (that is feature C1).
    supporting = [
        r
        for r in reads
        if r.has_sa_tag
        or r.max_clip >= min_clip_length
        or r.max_insertion >= min_sv_length
        or r.max_deletion >= min_sv_length
    ]
    n_support = len(supporting) or spanning_with_indel
    denom = n_support + clean_spanning
    out["support_ratio"] = (n_support / denom) if denom > 0 else None

    if supporting:
        n_fwd = sum(1 for r in supporting if not r.is_reverse)
        n_rev = len(supporting) - n_fwd
        total = n_fwd + n_rev
        out["strand_balance"] = abs(n_fwd - n_rev) / total if total else None
    else:
        out["strand_balance"] = None

    # -- local alignment quality ------------------------------------------------
    out["mean_read_len_local"] = safe_mean([r.aligned_length for r in reads])
    nm_rates = [r.nm_rate for r in reads if r.nm_rate is not None]
    out["mean_nm_rate"] = safe_mean(nm_rates) if nm_rates else None

    return out


def _depth_ratio(cand: CandidateRecord, bam: BamReader, w_flank: int) -> float | None:
    """Depth inside the SV span divided by mean flank depth.

    Structurally undefined for INS and BND, which consume no reference span -- returned
    as None rather than imputed, so the indicator column can carry that distinction
    (docs/03_feature_dictionary.md, missing-value strategy case 1).
    """
    if not cand.has_reference_span:
        return None

    inside = bam.mean_depth(cand.chrom, cand.pos, cand.end)
    left = bam.mean_depth(cand.chrom, max(0, cand.pos - w_flank), cand.pos)
    right = bam.mean_depth(cand.chrom, cand.end, cand.end + w_flank)

    flanks = [d for d in (left, right) if d > 0]
    if not flanks:
        return None
    flank_mean = sum(flanks) / len(flanks)
    if flank_mean <= 0:
        return None

    ratio = inside / flank_mean
    # Guard against pathological pileups in collapsed repeats producing absurd values.
    return float(min(ratio, 100.0)) if math.isfinite(ratio) else None
