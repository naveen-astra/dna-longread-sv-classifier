"""Candidate record types and SV-type normalisation.

Deliberately free of pysam so the schema, the feature registry and the pure logic
remain importable (and unit-testable) on hosts without the bioinformatics toolchain --
notably the Windows development machine, where pysam has no build (finding M4).
`svml.io.vcf` adds the pysam-dependent reading and writing on top of this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical SV types. BND is kept distinct from TRA: Sniffles2 emits BND records, and
# relabelling them TRA would assert a translocation the caller did not claim.
CANONICAL_TYPES = {"DEL", "INS", "DUP", "INV", "BND"}

_TYPE_ALIASES = {
    "DELETION": "DEL",
    "INSERTION": "INS",
    "DUPLICATION": "DUP",
    "DUP:TANDEM": "DUP",
    "DUP:INT": "DUP",
    "INVERSION": "INV",
    "TRA": "BND",
    "TRANSLOCATION": "BND",
    "BND": "BND",
}

# Supporting-read count is named differently by each caller and has changed across
# versions, so it is probed rather than assumed (spec section 9 stage 4).
SUPPORT_FIELD_ALIASES = ["SUPPORT", "RE", "RNAMES_COUNT", "DV", "SR"]


def normalise_svtype(raw: str | None) -> str | None:
    """Map a caller's SVTYPE string onto a canonical type, or None if unrecognised."""
    if raw is None:
        return None
    key = str(raw).upper().lstrip("<").rstrip(">")
    if key in CANONICAL_TYPES:
        return key
    return _TYPE_ALIASES.get(key)


@dataclass
class CandidateRecord:
    """One merged candidate SV. One of these becomes one row of the feature matrix."""

    candidate_id: str
    chrom: str
    pos: int
    end: int
    svtype: str
    svlen: int
    qual: float | None = None
    filter_pass: bool = True
    callers: list[str] = field(default_factory=list)
    per_caller: dict[str, dict[str, Any]] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def span_len(self) -> int:
        """Reference interval length. Zero for insertions, which consume no reference."""
        if self.svtype == "INS":
            return 0
        return max(0, self.end - self.pos)

    @property
    def has_reference_span(self) -> bool:
        """Whether a depth ratio across the event is even defined."""
        return self.svtype in {"DEL", "DUP", "INV"} and self.span_len > 0
