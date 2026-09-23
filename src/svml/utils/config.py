"""Configuration loading and validation.

No path or parameter is hardcoded in code (spec section 21); everything flows from
config/config.yaml, config/samples.tsv and config/references.yaml.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REQUIRED_SAMPLE_COLUMNS = [
    "sample_id",
    "platform",
    "preset",
    "role",
    "truth",
]

VALID_ROLES = {"labelled", "generalisation", "unlabelled"}
VALID_PLATFORMS = {"ONT", "HIFI"}


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or internally inconsistent."""


@dataclass(frozen=True)
class Config:
    """Validated run configuration.

    `raw` holds the parsed config.yaml. `samples` and `references` hold the two
    companion manifests. `digest` is a stable hash of all three, recorded alongside
    every results artefact so a metrics file can be tied back to the exact inputs
    that produced it (spec section 23).
    """

    raw: dict[str, Any]
    samples: pd.DataFrame
    references: dict[str, Any]
    digest: str
    root: Path

    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. `cfg.get("align.threads")`."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"required config key missing: {dotted}")
        return value

    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 42))

    @property
    def workdir(self) -> Path:
        return Path(self.require("project.workdir")).expanduser()

    def results(self, *parts: str) -> Path:
        return self.workdir.joinpath("results", *parts)

    def resources(self, *parts: str) -> Path:
        return self.workdir.joinpath("resources", *parts)

    # -- region / split helpers -------------------------------------------------

    def regions_for_role(self, role: str) -> list[str]:
        """samtools-style region strings to acquire for a sample role (finding M2).

        Tier 1 lists windows per role; tier 2 uses whole chromosomes for every role.
        """
        tier = int(self.get("regions.tier", 1))
        if tier == 1:
            tier1 = self.require("regions.tier1")
            if isinstance(tier1, list):  # legacy form: one chromosome list for all roles
                return list(tier1)
            return list(tier1.get(role, tier1.get("labelled", [])))
        tier2 = self.require("regions.tier2")
        return list(tier2["train"]) + list(tier2["val"]) + list(tier2["test"])

    def active_regions(self) -> list[str]:
        """Every region string in play for the current tier, across all roles."""
        seen: list[str] = []
        for role in ("labelled", "generalisation", "unlabelled"):
            for region in self.regions_for_role(role):
                if region not in seen:
                    seen.append(region)
        return seen

    def active_windows(self, role: str = "labelled") -> list[tuple[str, int, int]]:
        """Parsed (chrom, start, end) windows, 0-based half-open. Whole chroms -> (c, 0, inf)."""
        return [parse_region(r) for r in self.regions_for_role(role)]

    def active_chroms(self) -> list[str]:
        return sorted({parse_region(r)[0] for r in self.active_regions()})

    def split_map(self) -> dict[str, str]:
        """chromosome -> split name. Chromosome-grouped split (finding H3)."""
        tier2 = self.require("regions.tier2")
        mapping: dict[str, str] = {}
        for split in ("train", "val", "test"):
            for chrom in tier2[split]:
                if chrom in mapping:
                    raise ConfigError(
                        f"chromosome {chrom} appears in both {mapping[chrom]} and {split}; "
                        "splits must be disjoint to prevent leakage"
                    )
                mapping[chrom] = split
        return mapping

    def labelled_samples(self) -> list[str]:
        return self.samples.loc[self.samples["role"] == "labelled", "sample_id"].tolist()

    def all_samples(self) -> list[str]:
        return self.samples["sample_id"].tolist()

    def sample_row(self, sample_id: str) -> pd.Series:
        rows = self.samples.loc[self.samples["sample_id"] == sample_id]
        if rows.empty:
            raise ConfigError(f"unknown sample_id: {sample_id}")
        return rows.iloc[0]


WHOLE_CHROM_END = 10**12


def parse_region(region: str) -> tuple[str, int, int]:
    """Parse "chr16:1-15000000" (1-based, inclusive) into 0-based half-open coordinates.

    A bare chromosome name spans the whole chromosome.
    """
    if ":" not in region:
        return region, 0, WHOLE_CHROM_END
    chrom, span = region.split(":", 1)
    start, end = span.replace(",", "").split("-", 1)
    start_i, end_i = int(start), int(end)
    if start_i < 1 or end_i < start_i:
        raise ConfigError(f"invalid region: {region}")
    return chrom, start_i - 1, end_i


def in_windows(
    chrom: str, pos: int, windows: list[tuple[str, int, int]], margin: int = 0
) -> bool:
    """Whether a position falls inside any window, at least `margin` bp from its edges."""
    for w_chrom, w_start, w_end in windows:
        if chrom != w_chrom:
            continue
        lo = w_start + (margin if w_start > 0 else 0)
        hi = w_end - (margin if w_end < WHOLE_CHROM_END else 0)
        if lo <= pos < hi:
            return True
    return False


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at the top level of {path}")
    return data


def load_samples(path: Path) -> pd.DataFrame:
    """Load and validate the sample manifest.

    The manifest is a TSV with leading `#` comment lines.
    """
    if not path.exists():
        raise ConfigError(f"sample manifest not found: {path}")
    frame = pd.read_csv(path, sep="\t", comment="#", dtype=str).fillna("")
    frame.columns = [c.strip() for c in frame.columns]

    missing = [c for c in REQUIRED_SAMPLE_COLUMNS if c not in frame.columns]
    if missing:
        raise ConfigError(f"sample manifest missing columns: {missing}")

    if frame["sample_id"].duplicated().any():
        dupes = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ConfigError(f"duplicate sample_id values: {dupes}")

    bad_roles = set(frame["role"]) - VALID_ROLES
    if bad_roles:
        raise ConfigError(f"invalid role(s) {bad_roles}; expected one of {VALID_ROLES}")

    bad_platforms = set(frame["platform"]) - VALID_PLATFORMS
    if bad_platforms:
        raise ConfigError(
            f"invalid platform(s) {bad_platforms}; expected one of {VALID_PLATFORMS}"
        )

    if not (frame["role"] == "labelled").any():
        raise ConfigError("at least one sample must have role='labelled' to train a model")

    return frame


def load_references(path: Path) -> dict[str, Any]:
    refs = _read_yaml(path)
    for key in ("genome", "truth", "annotations"):
        if key not in refs:
            raise ConfigError(f"references.yaml missing required section: {key}")
    return refs


def _digest(*payloads: Any) -> str:
    blob = json.dumps(payloads, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_config(
    config_path: str | Path = "config/config.yaml",
    root: str | Path | None = None,
) -> Config:
    """Load config.yaml plus its two companion manifests, validating consistency."""
    config_path = Path(config_path)
    root_path = Path(root) if root is not None else config_path.resolve().parent.parent

    raw = _read_yaml(config_path)

    samples_path = root_path / raw.get("samples", {}).get("manifest", "config/samples.tsv")
    refs_path = root_path / raw.get("reference", {}).get("manifest", "config/references.yaml")

    samples = load_samples(samples_path)
    references = load_references(refs_path)

    cfg = Config(
        raw=raw,
        samples=samples,
        references=references,
        digest=_digest(raw, samples.to_dict("records"), references),
        root=root_path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Cross-manifest consistency checks that catch silent misconfiguration early."""
    # Splits must be disjoint. This is the leakage invariant (finding H3).
    cfg.split_map()

    # The labelled regions must reach train, val AND test chromosomes. The split is
    # grouped by chromosome, so a pilot on one chromosome leaves two splits empty and
    # nothing can be trained or evaluated.
    split_map = cfg.split_map()
    covered = {split_map.get(chrom) for chrom, _, _ in cfg.active_windows("labelled")}
    missing_splits = {"train", "val", "test"} - covered
    if missing_splits:
        raise ConfigError(
            f"labelled regions cover no {sorted(missing_splits)} chromosome(s). The split is "
            "chromosome-grouped, so every split needs at least one region or it is empty."
        )

    # Every sample's declared truth must exist in references.yaml, or be "none".
    known_truth = set(cfg.references.get("truth", {}).keys()) | {"none", ""}
    for _, row in cfg.samples.iterrows():
        if row["truth"] not in known_truth:
            raise ConfigError(
                f"sample {row['sample_id']} declares truth='{row['truth']}' "
                f"which is not defined in references.yaml (known: {sorted(known_truth)})"
            )
        if row["role"] in {"labelled", "generalisation"} and row["truth"] in {"none", ""}:
            raise ConfigError(
                f"sample {row['sample_id']} has role='{row['role']}' but no truth set; "
                "only role='unlabelled' may omit truth"
            )

    # The labelling matcher and the evaluation matcher must agree, or the reported
    # metrics measure the mismatch rather than the model (finding H5).
    label_params = cfg.get("labels.truvari", {})
    eval_params = cfg.get("evaluation.truvari", {})
    if label_params != eval_params:
        raise ConfigError(
            "labels.truvari and evaluation.truvari must be identical -- using different "
            "match criteria for labelling and benchmarking invalidates the comparison. "
            f"labels={label_params} evaluation={eval_params}"
        )

    # The confident-region gate is what makes the negative class trustworthy (H4).
    if not cfg.get("labels.require_inside_confident_regions", True):
        raise ConfigError(
            "labels.require_inside_confident_regions must be true. Disabling the gate "
            "labels unknown candidates as false positives, which teaches the model that "
            "real SVs in hard regions are artefacts -- inverting the project's goal."
        )

    # Stage C is pseudo-labelled and must never be marked benchmarked (finding B1).
    if cfg.get("ml.stages.C.benchmarked", False):
        raise ConfigError(
            "ml.stages.C.benchmarked must be false. No GIAB truth set labels DUP/INV/TRA, "
            "so five-class results rest on caller-consensus pseudo-labels and cannot be "
            "reported as benchmarked ground truth (see docs/00_architecture_evaluation.md B1)."
        )

    # Stage B must exclude the caller's declared type, or it predicts its own input.
    excluded = set(cfg.get("ml.stages.B.exclude_features", []))
    if "svtype_called" not in excluded:
        raise ConfigError(
            "ml.stages.B.exclude_features must contain 'svtype_called'; otherwise the "
            "DEL/INS classifier learns the identity function (finding H1)."
        )

    # The baseline set must include the two comparisons that make the claim honest (H2).
    baselines = set(cfg.get("evaluation.baselines", []))
    for required in ("union", "intersection"):
        if required not in baselines:
            raise ConfigError(
                f"evaluation.baselines must include '{required}'. Without it the ML layer "
                "is not compared against the trivial heuristic it must beat (finding H2)."
            )
