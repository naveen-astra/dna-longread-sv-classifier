"""Provenance capture (spec section 23).

Records tool versions, reference build, dataset versions, config hash, random seeds,
model hyperparameters and execution date so a results file can be traced back to the
exact inputs and software that produced it.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from svml.utils.config import Config
from svml.utils.logging import get_logger

log = get_logger(__name__)

# tool -> command that prints its version
TOOL_VERSION_COMMANDS: dict[str, list[str]] = {
    "minimap2": ["minimap2", "--version"],
    "samtools": ["samtools", "--version"],
    "bcftools": ["bcftools", "--version"],
    "bedtools": ["bedtools", "--version"],
    "sniffles": ["sniffles", "--version"],
    "cuteSV": ["cuteSV", "--version"],
    "SURVIVOR": ["SURVIVOR", "--version"],
    "truvari": ["truvari", "version"],
    "NanoPlot": ["NanoPlot", "--version"],
}

PYTHON_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "shap",
    "pysam",
    "matplotlib",
    "seaborn",
    "snakemake",
]


def _run_version(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - env dependent
        log.warning("could not get version for %s: %s", cmd[0], exc)
        return None
    # Some tools print their version to stderr rather than stdout.
    output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    return output.splitlines()[0].strip() if output else None


def tool_versions() -> dict[str, str | None]:
    """Version string per external tool. `None` means the tool was not found on PATH."""
    return {name: _run_version(cmd) for name, cmd in TOOL_VERSION_COMMANDS.items()}


def package_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for pkg in PYTHON_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def missing_tools() -> list[str]:
    """External tools not present on PATH. Used by the Phase 0 exit test."""
    return [name for name, cmd in TOOL_VERSION_COMMANDS.items() if shutil.which(cmd[0]) is None]


def build_metadata(cfg: Config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    genome = cfg.references.get("genome", {})
    truth_key = cfg.get("labels.truth")
    truth = cfg.references.get("truth", {}).get(truth_key, {})

    meta: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "svml_version": __import__("svml").__version__,
        "config_digest": cfg.digest,
        "random_seed": cfg.seed,
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "reference": {
            "id": genome.get("id"),
            "build": genome.get("build"),
            "contig_naming": genome.get("contig_naming"),
        },
        "truth": {
            "key": truth_key,
            "release": truth.get("release"),
            "build": truth.get("build"),
            "sample": truth.get("sample"),
        },
        "region_tier": cfg.get("regions.tier"),
        "active_regions": cfg.active_regions(),
        "split_map": cfg.split_map(),
        "samples": cfg.samples.to_dict("records"),
        "match_criteria": cfg.get("labels.truvari", {}),
        "tool_versions": tool_versions(),
        "package_versions": package_versions(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_metadata(cfg: Config, out_path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = build_metadata(cfg, extra)
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log.info("wrote provenance metadata to %s", path)
    return path
