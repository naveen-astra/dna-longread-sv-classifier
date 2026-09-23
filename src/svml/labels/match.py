"""Truth matching via Truvari (finding H5).

`truvari bench` is used for BOTH label creation and benchmarking, with identical
parameters enforced by `svml.utils.config`. If the labeller and the evaluator disagreed
about what counts as a match, the reported metrics would measure that disagreement
rather than the model.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from svml.io.records import normalise_svtype
from svml.utils.logging import get_logger

log = get_logger(__name__)


class TruvariError(RuntimeError):
    pass


@dataclass
class TruvariParams:
    refdist: int = 500
    pctseq: float = 0.7
    pctsize: float = 0.7
    pctovl: float = 0.0
    refine: bool = True
    passonly: bool = True
    sizemin: int = 50
    extra: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, params: dict[str, Any]) -> TruvariParams:
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        return cls(**{k: v for k, v in params.items() if k in known})

    def to_args(self) -> list[str]:
        args = [
            "--refdist", str(self.refdist),
            "--pctseq", str(self.pctseq),
            "--pctsize", str(self.pctsize),
            "--pctovl", str(self.pctovl),
            "--sizemin", str(self.sizemin),
        ]
        if self.passonly:
            args.append("--passonly")
        args.extend(self.extra)
        return args


def run_bench(
    comparison_vcf: str | Path,
    truth_vcf: str | Path,
    confident_bed: str | Path,
    out_dir: str | Path,
    params: TruvariParams,
    reference_fasta: str | Path | None = None,
) -> Path:
    """Run `truvari bench`, optionally followed by `truvari refine`.

    The v5.0q benchmark README recommends the refine step, because that draft benchmark
    does not exclude complex SVs and naive comparison mistakes representation
    differences for real errors.
    """
    if shutil.which("truvari") is None:
        raise TruvariError(
            "truvari not found on PATH. Truth matching and benchmarking must use a "
            "standard tool rather than a bespoke matcher (finding H5)."
        )

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Invoked as `python -m svml._truvari_entry` rather than the `truvari` binary
    # directly: that entry point patches truvari.compress_index_vcf, which otherwise
    # crashes on output finalization in this environment (verified reproducible across
    # truvari 4.2.2/4.3.1/5.4.0 and pysam 0.22.1/0.24.1 -- see _truvari_entry.py and
    # docs/06_execution_plan.md, deviation D11). The actual comparison engine is
    # unaffected; only the bgzip/index finalization step needed the workaround.
    cmd = [
        sys.executable, "-m", "svml._truvari_entry", "bench",
        "-b", str(truth_vcf),
        "-c", str(comparison_vcf),
        "--includebed", str(confident_bed),
        "-o", str(out_dir),
        *params.to_args(),
    ]
    log.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise TruvariError(f"truvari bench failed ({proc.returncode}): {proc.stderr.strip()}")

    if params.refine:
        if reference_fasta is None:
            log.warning("refine requested but no reference FASTA supplied; skipping refine step")
        else:
            refine_cmd = [
                sys.executable, "-m", "svml._truvari_entry", "refine",
                "--reference", str(reference_fasta),
                "--regions", str(out_dir / "candidate.refine.bed"),
                str(out_dir),
            ]
            log.info("running: %s", " ".join(refine_cmd))
            refine = subprocess.run(refine_cmd, capture_output=True, text=True, check=False)
            if refine.returncode != 0:
                # A refine failure is not fatal -- the unrefined result is still usable,
                # but the report must say the refine step did not run.
                log.warning("truvari refine failed: %s", refine.stderr.strip())

    return out_dir


def read_summary(bench_dir: str | Path) -> dict[str, Any]:
    """Parse truvari's summary.json into a plain dict."""
    path = Path(bench_dir) / "summary.json"
    if not path.exists():
        raise TruvariError(f"truvari summary not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_ids(vcf_path: Path, svtype_from_truth: bool) -> dict[str, str | None]:
    """Map candidate ID -> matched SVTYPE (or None) from a truvari output VCF.

    truvari's tp-comp.vcf.gz annotates match-quality metrics (PctSeqSimilarity,
    TruScore, MatchId, ...) but does not carry the matched truth record's own SVTYPE
    under any field name -- verified directly against real truvari 4.3.1 output, not
    assumed. Reading the CANDIDATE's own declared SVTYPE from that same record is a
    valid substitute: truvari's default matching requires type compatibility (this
    project's config never passes --typeignore), so a record only appears in
    tp-comp.vcf.gz at all if its type already matched the truth's.

    `rec.info.get(key)` raises ValueError (not KeyError) for a key absent from the
    VCF header entirely, as opposed to declared-but-unset; both are treated as
    "no value" here.
    """
    import pysam

    if not vcf_path.exists():
        return {}

    out: dict[str, str | None] = {}
    with pysam.VariantFile(str(vcf_path)) as vcf:
        for rec in vcf:
            if rec.id is None:
                continue
            svtype = None
            if svtype_from_truth:
                try:
                    raw = rec.info.get("SVTYPE")
                except (KeyError, ValueError):
                    raw = None
                if isinstance(raw, tuple):
                    raw = raw[0] if raw else None
                svtype = normalise_svtype(raw)
            out[str(rec.id)] = svtype
    return out


def labels_from_bench(
    bench_dir: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Assign `is_true` and `true_sv_type` from a truvari bench result.

    Only candidates already marked `inside_confident` by the gate are labelled. Rows
    outside confident regions keep `is_true = NA` and are excluded from train and test.
    """
    bench_dir = Path(bench_dir)
    tp_ids = _collect_ids(bench_dir / "tp-comp.vcf.gz", svtype_from_truth=True)
    fp_ids = _collect_ids(bench_dir / "fp.vcf.gz", svtype_from_truth=False)

    log.info("truvari matched %d TP and %d FP candidates", len(tp_ids), len(fp_ids))

    frame = frame.copy()
    if "label_status" not in frame.columns:
        raise ValueError(
            "apply_gate() must run before labels_from_bench(); the confident-region gate "
            "is what makes the negative class trustworthy (finding H4)"
        )

    inside = frame["label_status"] == "inside_confident"

    is_true: list[float | None] = []
    true_type: list[str | None] = []
    for row in frame.itertuples(index=False):
        cid = str(row.candidate_id)
        if not (row.label_status == "inside_confident"):
            is_true.append(None)
            true_type.append(None)
        elif cid in tp_ids:
            is_true.append(1.0)
            true_type.append(tp_ids[cid] or row.svtype)
        elif cid in fp_ids:
            is_true.append(0.0)
            true_type.append(None)
        else:
            # Inside a confident region but absent from both truvari outputs. This
            # happens when truvari's own region filtering differs slightly from the
            # gate. Treated as unknown rather than guessed either way.
            is_true.append(None)
            true_type.append(None)

    frame["is_true"] = is_true
    frame["true_sv_type"] = true_type

    unresolved = int(inside.sum() - frame.loc[inside, "is_true"].notna().sum())
    if unresolved:
        log.warning(
            "%d candidates inside confident regions were not resolved by truvari and "
            "remain unlabelled",
            unresolved,
        )
    return frame


def count_missed_truth(
    bench_dir: str | Path,
    windows: list[tuple[str, int, int]],
    margin: int = 0,
) -> dict[str, int]:
    """Truth SVs that no candidate matched, per chromosome, inside the evaluated windows.

    These are truvari's unmatched base records (fn.vcf.gz), already restricted to the
    confident regions by --includebed. They are false negatives for every method --
    baselines and ML alike -- and without them recall would be measured against the
    candidate set instead of against the truth.
    """
    import pysam

    from svml.utils.config import in_windows

    path = Path(bench_dir) / "fn.vcf.gz"
    if not path.exists():
        log.warning("truvari fn.vcf.gz not found at %s; missed-truth count unavailable", path)
        return {}

    counts: dict[str, int] = {}
    with pysam.VariantFile(str(path)) as vcf:
        for rec in vcf:
            if in_windows(rec.chrom, int(rec.pos), windows, margin):
                counts[rec.chrom] = counts.get(rec.chrom, 0) + 1
    log.info("truth SVs missed by both callers, per chromosome: %s", counts)
    return counts


def add_pseudo_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Caller-consensus pseudo-labels for stage C (finding B1).

    NOT GROUND TRUTH. No GIAB truth set labels DUP, INV or TRA, so stage C measures
    agreement with the callers' consensus -- a much weaker quantity than accuracy.
    Every stage C number must be reported with that provenance attached.
    """
    frame = frame.copy()

    def pseudo(row) -> str | None:
        # Only assign a pseudo-label where both callers agree on the type.
        if row.get("n_callers", 0) < 2:
            return None
        svtype = row.get("svtype")
        # BND is reported as TRA at the presentation layer only; the callers emit BND.
        return "TRA" if svtype == "BND" else svtype

    frame["pseudo_sv_type"] = frame.apply(pseudo, axis=1)

    # Where real labels exist they take precedence over the pseudo-label, so stage C
    # is at least anchored to truth for DEL and INS.
    if "true_sv_type" in frame.columns:
        real = frame["true_sv_type"].notna()
        frame.loc[real, "pseudo_sv_type"] = frame.loc[real, "true_sv_type"]

    counts = frame["pseudo_sv_type"].value_counts(dropna=True).to_dict()
    log.info("stage C pseudo-label distribution (NOT ground truth): %s", counts)
    return frame
