"""I/O layer.

`records` and `bed` are pure python and always importable. `vcf` and `bam` need pysam,
which has no native Windows build, so they are imported on demand rather than eagerly --
that keeps the schema and the pure logic testable on the development machine
(finding M4).
"""

from svml.io.bed import IntervalIndex, check_contig_naming, load_bed
from svml.io.records import (
    CANONICAL_TYPES,
    SUPPORT_FIELD_ALIASES,
    CandidateRecord,
    normalise_svtype,
)

__all__ = [
    "IntervalIndex",
    "check_contig_naming",
    "load_bed",
    "CANONICAL_TYPES",
    "SUPPORT_FIELD_ALIASES",
    "CandidateRecord",
    "normalise_svtype",
    "read_candidates",
    "write_annotated_vcf",
    "BamReader",
]

_LAZY = {
    "read_candidates": "svml.io.vcf",
    "write_annotated_vcf": "svml.io.vcf",
    "describe_info_fields": "svml.io.vcf",
    "BamReader": "svml.io.bam",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'svml.io' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
