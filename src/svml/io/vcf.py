"""VCF reading and writing for candidate structural variants.

Requires pysam. The pysam-free record types and type normalisation live in
`svml.io.records`, so the schema and pure logic stay importable without the
bioinformatics toolchain.

Spec section 9 stage 4 requires that INFO field names be inspected from real tool
output rather than assumed, so this module probes several known aliases per field and
records which one was found.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from svml.io.records import (
    CANONICAL_TYPES,
    SUPPORT_FIELD_ALIASES,
    CandidateRecord,
    normalise_svtype,
)
from svml.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import pysam

log = get_logger(__name__)

__all__ = [
    "CANONICAL_TYPES",
    "SUPPORT_FIELD_ALIASES",
    "CandidateRecord",
    "normalise_svtype",
    "compute_end",
    "compute_svlen",
    "describe_info_fields",
    "extract_support",
    "read_candidates",
    "write_annotated_vcf",
]


def _pysam():
    """Import pysam on demand, with an actionable message when it is unavailable."""
    try:
        import pysam
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pysam is required for VCF/BAM access but is not installed. It has no native "
            "Windows build -- run inside WSL2 or Docker with the svml conda environment "
            "activated (see docs/06_execution_plan.md, Phase 0)."
        ) from exc
    return pysam


def _info_get(rec: pysam.VariantRecord, key: str, default: Any = None) -> Any:
    try:
        value = rec.info.get(key, default)
    except (KeyError, ValueError):
        return default
    if isinstance(value, tuple):
        return value[0] if value else default
    return value


def extract_support(rec: pysam.VariantRecord) -> tuple[int | None, str | None]:
    """Supporting-read count plus the INFO field it came from.

    Returns (None, None) when no known alias is present, rather than guessing zero --
    absent evidence and zero evidence are different things (see the missing-value
    strategy in docs/03_feature_dictionary.md).
    """
    for alias in SUPPORT_FIELD_ALIASES:
        value = _info_get(rec, alias)
        if value is not None:
            try:
                return int(value), alias
            except (TypeError, ValueError):
                continue

    if rec.samples:
        sample = rec.samples[0]
        for alias in ("DV", "DR"):
            if alias in sample and sample[alias] is not None:
                value = sample[alias]
                value = value[0] if isinstance(value, tuple) else value
                try:
                    return int(value), f"FORMAT/{alias}"
                except (TypeError, ValueError):
                    continue
    return None, None


def compute_svlen(rec: pysam.VariantRecord, svtype: str) -> int:
    """Resolve SVLEN, falling back to END and then to allele length difference."""
    svlen = _info_get(rec, "SVLEN")
    if svlen is not None:
        try:
            return int(svlen)
        except (TypeError, ValueError):
            pass

    end = _info_get(rec, "END")
    if end is not None and svtype in {"DEL", "DUP", "INV"}:
        try:
            return int(end) - int(rec.pos)
        except (TypeError, ValueError):
            pass

    if rec.alts and rec.ref and not str(rec.alts[0]).startswith("<"):
        return len(str(rec.alts[0])) - len(str(rec.ref))
    return 0


def compute_end(rec: pysam.VariantRecord, svtype: str, svlen: int) -> int:
    """Resolve END, preferring the explicit INFO field over inference."""
    end = _info_get(rec, "END")
    if end is not None:
        try:
            return int(end)
        except (TypeError, ValueError):
            pass
    if rec.stop is not None and rec.stop > rec.pos:
        return int(rec.stop)
    if svtype == "INS":
        return int(rec.pos)
    return int(rec.pos) + abs(int(svlen))


def describe_info_fields(vcf_path: str | Path) -> dict[str, str]:
    """INFO field id -> description, for the Phase 6 header inspection exit test."""
    pysam = _pysam()
    with pysam.VariantFile(str(vcf_path)) as vcf:
        return {rec.name: (rec.description or "") for rec in vcf.header.info.values()}


def read_candidates(
    vcf_path: str | Path,
    min_sv_length: int = 50,
    caller: str | None = None,
    restrict_chroms: set[str] | None = None,
) -> list[CandidateRecord]:
    """Read a candidate VCF into CandidateRecord objects.

    Records shorter than `min_sv_length` are dropped (project scope: SVs >= 50 bp).
    BND records are kept regardless of SVLEN, since a breakend has no meaningful
    length; stages that need a reference span filter them out later.
    """
    pysam = _pysam()
    path = Path(vcf_path)
    candidates: list[CandidateRecord] = []
    skipped_type = skipped_len = 0

    with pysam.VariantFile(str(path)) as vcf:
        for i, rec in enumerate(vcf):
            svtype = normalise_svtype(_info_get(rec, "SVTYPE"))
            if svtype is None and rec.alts:
                svtype = normalise_svtype(str(rec.alts[0]))
            if svtype is None:
                skipped_type += 1
                continue

            if restrict_chroms is not None and rec.chrom not in restrict_chroms:
                continue

            svlen = compute_svlen(rec, svtype)
            if svtype != "BND" and abs(svlen) < min_sv_length:
                skipped_len += 1
                continue

            end = compute_end(rec, svtype, svlen)
            support, support_field = extract_support(rec)
            filter_pass = ("PASS" in rec.filter.keys()) or (len(rec.filter.keys()) == 0)

            caller_name = caller or path.stem
            candidates.append(
                CandidateRecord(
                    candidate_id=str(rec.id or f"{path.stem}_{i}"),
                    chrom=rec.chrom,
                    pos=int(rec.pos),
                    end=int(end),
                    svtype=svtype,
                    svlen=int(svlen),
                    qual=rec.qual,
                    filter_pass=filter_pass,
                    callers=[caller_name],
                    per_caller={
                        caller_name: {
                            "support": support,
                            "support_field": support_field,
                            "qual": rec.qual,
                            "filter_pass": filter_pass,
                            "pos": int(rec.pos),
                            "svlen": int(svlen),
                            "svtype": svtype,
                        }
                    },
                )
            )

    log.info(
        "read %d candidates from %s (skipped %d untyped, %d below %d bp)",
        len(candidates), path.name, skipped_type, skipped_len, min_sv_length,
    )
    return candidates


def write_annotated_vcf(
    template_vcf: str | Path,
    out_path: str | Path,
    predictions: dict[str, dict[str, Any]],
    filter_threshold: float | None = None,
) -> Path:
    """Write the final VCF, annotating each record with the model's score.

    `predictions` maps candidate_id -> {"score", "predicted_type", "label_status"}.
    Records scoring below `filter_threshold` are marked FILTER=SVML_LOWCONF rather than
    removed, so nothing is silently discarded.
    """
    pysam = _pysam()
    template_vcf = Path(template_vcf)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pysam.VariantFile(str(template_vcf)) as src:
        # Modify src.header IN PLACE and write each `rec` directly, rather than
        # `header.copy()` + `rec.translate(header)`. Root-caused directly on real
        # merged-candidate output (finding D19): `.translate()` against a *copied*
        # header returned None universally -- for every record, including plain
        # DEL/INS, not just BND -- while setting new INFO fields on the original
        # record and writing it against the header it was already bound to (with
        # fields added) works cleanly for every record type. This is the documented
        # pysam idiom for adding annotations to an existing VCF; the copy+translate
        # path this replaced was the actual defect, not any property of the data.
        header = src.header
        for item in (
            [("ID", "SVML_SCORE"), ("Number", "1"), ("Type", "Float"),
             ("Description", "svml calibrated P(true) for this candidate")],
            [("ID", "SVML_TYPE"), ("Number", "1"), ("Type", "String"),
             ("Description", "svml predicted SV type")],
            [("ID", "SVML_LABEL_STATUS"), ("Number", "1"), ("Type", "String"),
             ("Description", "Benchmark label status: inside_confident or "
                             "outside_confident (unscored by the benchmark)")],
        ):
            header.add_meta("INFO", items=item)
        header.add_meta(
            "FILTER",
            items=[("ID", "SVML_LOWCONF"),
                   ("Description", f"svml score below threshold {filter_threshold}")],
        )

        mode = "wz" if str(out_path).endswith(".gz") else "w"
        written = 0
        with pysam.VariantFile(str(out_path), mode, header=header) as dst:
            for rec in src:
                pred = predictions.get(str(rec.id))
                if pred is not None:
                    score = pred.get("score")
                    if score is not None:
                        rec.info["SVML_SCORE"] = float(score)
                        if filter_threshold is not None and float(score) < filter_threshold:
                            rec.filter.add("SVML_LOWCONF")
                    if pred.get("predicted_type"):
                        rec.info["SVML_TYPE"] = str(pred["predicted_type"])
                    if pred.get("label_status"):
                        rec.info["SVML_LABEL_STATUS"] = str(pred["label_status"])
                dst.write(rec)
                written += 1

    if str(out_path).endswith(".gz"):
        pysam.tabix_index(str(out_path), preset="vcf", force=True)

    log.info("wrote %d annotated records to %s", written, out_path)
    return out_path
