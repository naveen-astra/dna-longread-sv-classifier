"""Candidate merging across callers.

Finding H5: merging is delegated to a community-standard tool, then provenance (which
caller supported which candidate) is reconciled back against the per-caller VCFs.

Default tool: `truvari collapse`. SURVIVOR is kept as an option but is NOT the default,
because SURVIVOR emits symbolic alleles (<DEL>, <INS>), and Truvari -- used for labels
and benchmarking -- silently filters unresolved SVs whenever sequence similarity is
required (truvari/matching.py: "Unresolved SVs (e.g. ALT=<DEL>) are filtered when
`--pctseq != 0`"). A SURVIVOR-merged candidate set would therefore produce zero labels.
`truvari collapse` keeps the callers' sequence-resolved alleles, and makes merging and
labelling use the same matching engine.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from svml.io.records import CandidateRecord
from svml.io.vcf import read_candidates
from svml.utils.logging import get_logger

log = get_logger(__name__)


class MergeError(RuntimeError):
    pass


def run_survivor(
    vcf_paths: list[str | Path],
    out_vcf: str | Path,
    max_distance: int = 500,
    min_callers: int = 1,
    require_same_type: bool = True,
    require_same_strand: bool = False,
    min_sv_length: int = 50,
) -> Path:
    """Run `SURVIVOR merge`. Raises MergeError if SURVIVOR is not on PATH."""
    if shutil.which("SURVIVOR") is None:
        raise MergeError(
            "SURVIVOR not found on PATH. Merging must use a standard tool rather than a "
            "bespoke matcher (finding H5). Install it, or run inside the svml conda env."
        )

    out_vcf = Path(out_vcf)
    out_vcf.parent.mkdir(parents=True, exist_ok=True)

    # SURVIVOR only reads plain-text VCF, so decompress any bgzipped inputs first.
    workdir = Path(tempfile.mkdtemp(prefix="svml_survivor_", dir=out_vcf.parent))
    plain_inputs = [_as_plain_vcf(Path(p), workdir) for p in vcf_paths]

    list_file = str(workdir / "inputs.txt")
    Path(list_file).write_text("".join(f"{p.resolve()}\n" for p in plain_inputs))

    cmd = [
        "SURVIVOR", "merge", list_file,
        str(max_distance),
        str(min_callers),
        "1" if require_same_type else "0",
        "1" if require_same_strand else "0",
        "0",                      # estimate distance from SV size: off, use fixed distance
        str(min_sv_length),
        str(out_vcf),
    ]
    log.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    shutil.rmtree(workdir, ignore_errors=True)

    if proc.returncode != 0:
        raise MergeError(f"SURVIVOR merge failed ({proc.returncode}): {proc.stderr.strip()}")
    if not out_vcf.exists():
        raise MergeError(f"SURVIVOR reported success but {out_vcf} was not created")

    n = assign_unique_ids(out_vcf)
    log.info("SURVIVOR merge wrote %s (%d records, unique IDs assigned)", out_vcf, n)
    return out_vcf


def run_truvari_collapse(
    vcf_paths: dict[str, str | Path],
    out_vcf: str | Path,
    refdist: int = 500,
    pctseq: float = 0.7,
    pctsize: float = 0.7,
    min_sv_length: int = 50,
) -> Path:
    """Merge caller VCFs with `truvari collapse`, keeping sequence-resolved alleles.

    `vcf_paths` maps caller name -> bgzipped, indexed VCF. Each caller's sample column is
    renamed to a common name so `bcftools concat` accepts them, IDs are prefixed with the
    caller name, and the concatenation is collapsed so redundant calls of the same event
    reduce to one candidate.
    """
    for tool in ("truvari", "bcftools", "tabix"):
        if shutil.which(tool) is None:
            raise MergeError(f"{tool} not found on PATH; run inside the svml environment")

    out_vcf = Path(out_vcf)
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="svml_collapse_", dir=out_vcf.parent))

    def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
        log.info("running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)
        if proc.returncode != 0:
            raise MergeError(f"{cmd[0]} {cmd[1]} failed ({proc.returncode}): {proc.stderr.strip()}")
        return proc

    try:
        prepared = []
        for caller, path in vcf_paths.items():
            samples_file = workdir / f"{caller}.samples.txt"
            samples_file.write_text("SAMPLE\n")
            renamed = workdir / f"{caller}.renamed.vcf.gz"
            # One sample column named SAMPLE, first sample only, so concat accepts both.
            first = run(["bcftools", "query", "-l", str(path)]).stdout.split()
            view = ["bcftools", "view", "-Oz", "-o", str(workdir / f"{caller}.one.vcf.gz")]
            if first:
                view += ["-s", first[0]]
            run(view + [str(path)])
            run(["bcftools", "reheader", "-s", str(samples_file), "-o", str(renamed),
                 str(workdir / f"{caller}.one.vcf.gz")])
            run(["tabix", "-f", "-p", "vcf", str(renamed)])
            prepared.append(str(renamed))

        concat = workdir / "concat.vcf.gz"
        run(["bcftools", "concat", "-a", "-Oz", "-o", str(workdir / "concat.unsorted.vcf.gz"),
             *prepared])
        run(["bcftools", "sort", "-Oz", "-o", str(concat), str(workdir / "concat.unsorted.vcf.gz")])
        run(["tabix", "-f", "-p", "vcf", str(concat)])

        # Invoked via svml._truvari_entry, not the truvari binary directly: it patches
        # a broken finalization call in this environment. See its module docstring and
        # docs/06_execution_plan.md deviation D11.
        run([
            sys.executable, "-m", "svml._truvari_entry", "collapse",
            "-i", str(concat),
            "-o", str(out_vcf),
            "-c", str(workdir / "collapsed_away.vcf"),
            "--refdist", str(refdist),
            "--pctseq", str(pctseq),
            "--pctsize", str(pctsize),
            "--sizemin", str(min_sv_length),
        ])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if not out_vcf.exists():
        raise MergeError(f"truvari collapse reported success but {out_vcf} was not created")

    n = assign_unique_ids(out_vcf)
    log.info("truvari collapse wrote %s (%d records, unique IDs assigned)", out_vcf, n)
    return out_vcf


def _as_plain_vcf(path: Path, workdir: Path) -> Path:
    if not str(path).endswith(".gz"):
        return path
    import gzip

    target = workdir / path.name[: -len(".gz")]
    with gzip.open(path, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def assign_unique_ids(vcf_path: str | Path, prefix: str = "SVML") -> int:
    """Rewrite the ID column so every merged candidate has a unique, stable ID.

    Truvari reports matches by record ID, and labels are joined back to the feature
    matrix on that ID. SURVIVOR copies IDs from its inputs, which can be "." or collide
    across callers -- either would silently attach the wrong label to a candidate.
    """
    path = Path(vcf_path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, n = [], 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        fields = line.rstrip("\n").split("\t")
        n += 1
        fields[2] = f"{prefix}{n:07d}"
        out.append("\t".join(fields) + "\n")
    path.write_text("".join(out), encoding="utf-8")
    return n


def attach_provenance(
    merged_vcf: str | Path,
    per_caller_vcfs: dict[str, str | Path],
    max_distance: int = 500,
    min_sv_length: int = 50,
    restrict_chroms: set[str] | None = None,
) -> list[CandidateRecord]:
    """Reconcile merged candidates against per-caller calls to recover provenance.

    Needed because features C1-C5 (`sniffles_support`, `cutesv_support`, `n_callers`,
    `bp_offset_between_callers`, `svlen_disagreement`) all require knowing which caller
    contributed what.
    """
    merged = read_candidates(
        merged_vcf, min_sv_length=min_sv_length, caller="merged", restrict_chroms=restrict_chroms
    )

    per_caller: dict[str, list[CandidateRecord]] = {}
    for caller, path in per_caller_vcfs.items():
        per_caller[caller] = read_candidates(
            path, min_sv_length=min_sv_length, caller=caller, restrict_chroms=restrict_chroms
        )

    # Bucket each caller's calls by (chrom, type) so matching is linear rather than
    # quadratic across the whole callset.
    buckets: dict[str, dict[tuple[str, str], list[CandidateRecord]]] = {}
    for caller, records in per_caller.items():
        bucket: dict[tuple[str, str], list[CandidateRecord]] = {}
        for rec in records:
            bucket.setdefault((rec.chrom, rec.svtype), []).append(rec)
        for key in bucket:
            bucket[key].sort(key=lambda r: r.pos)
        buckets[caller] = bucket

    for cand in merged:
        cand.per_caller = {}
        cand.callers = []
        for caller, bucket in buckets.items():
            hit = _nearest(bucket.get((cand.chrom, cand.svtype), []), cand.pos, max_distance)
            if hit is not None:
                cand.callers.append(caller)
                cand.per_caller[caller] = hit.per_caller[caller]

        if not cand.per_caller:
            log.debug("merged candidate %s matched no per-caller record", cand.candidate_id)

    n_two = sum(1 for c in merged if len(c.per_caller) >= 2)
    log.info(
        "attached provenance: %d merged candidates, %d supported by both callers (%.1f%%)",
        len(merged),
        n_two,
        100 * n_two / max(1, len(merged)),
    )
    return merged


def _nearest(
    records: list[CandidateRecord], pos: int, max_distance: int
) -> CandidateRecord | None:
    best: CandidateRecord | None = None
    best_distance = max_distance + 1
    for rec in records:
        distance = abs(rec.pos - pos)
        if distance < best_distance:
            best, best_distance = rec, distance
    return best if best_distance <= max_distance else None


def merge_summary(candidates: list[CandidateRecord]) -> dict[str, int]:
    """Counts for the Phase 6 exit test."""
    summary = {
        "n_candidates": len(candidates),
        "n_single_caller": sum(1 for c in candidates if len(c.per_caller) == 1),
        "n_both_callers": sum(1 for c in candidates if len(c.per_caller) >= 2),
        "n_orphan": sum(1 for c in candidates if not c.per_caller),
    }
    for svtype in ("DEL", "INS", "DUP", "INV", "BND"):
        summary[f"n_{svtype}"] = sum(1 for c in candidates if c.svtype == svtype)
    return summary
