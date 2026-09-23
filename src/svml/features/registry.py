"""Feature registry -- the single source of truth for the feature schema.

docs/03_feature_dictionary.md is generated from this module, so the documentation and
the extraction code cannot drift apart (see docs/02_repository_architecture.md).

Groups exist to support the required ablation (finding M3):
    B  BAM-derived        -- independent of the callers
    A  annotation-derived -- independent of the callers
    C  caller-reported    -- the callers' own evidence fields
    V  caller-declared    -- the callers' declared type and length
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Group(str, Enum):
    BAM = "B"
    ANNOT = "A"
    CALLER = "C"
    DECLARED = "V"


class Missing(str, Enum):
    ZERO = "zero"  # genuinely zero: a real measurement, not missing data
    NAN_INDICATOR = "nan_indicator"  # structurally undefined: NaN + boolean indicator
    SENTINEL = "sentinel"  # large sentinel value (distance features)


@dataclass(frozen=True)
class Feature:
    name: str
    group: Group
    dtype: str
    definition: str
    missing: Missing
    meaning: str
    min_value: float | None = None
    max_value: float | None = None
    categorical: bool = False

    @property
    def needs_indicator(self) -> bool:
        return self.missing is Missing.NAN_INDICATOR


SENTINEL_DISTANCE = 1_000_000_000

# --------------------------------------------------------------------------------
# Group B -- BAM-derived
# --------------------------------------------------------------------------------
_GROUP_B = [
    Feature(
        "dp_local", Group.BAM, "float",
        "Mean per-base depth in +/- w_bp around the breakpoint.",
        Missing.ZERO,
        "Low depth means weak evidence for any call; very high depth flags collapsed repeats.",
        min_value=0.0,
    ),
    Feature(
        "dp_flank_ratio", Group.BAM, "float",
        "Mean depth inside the SV span divided by mean depth of the two flanks.",
        Missing.NAN_INDICATOR,
        "Direct copy-number signal: near 0 or 0.5 for a true DEL, above 1.3 for a true DUP. "
        "Undefined for INS and BND, which have no reference span.",
        min_value=0.0,
    ),
    Feature(
        "dp_mad_local", Group.BAM, "float",
        "Median absolute deviation of per-base depth in the breakpoint window.",
        Missing.ZERO,
        "Erratic local depth indicates mapping instability, common at artefactual calls.",
        min_value=0.0,
    ),
    Feature(
        "mean_mapq", Group.BAM, "float",
        "Mean mapping quality of reads overlapping the breakpoint.",
        Missing.NAN_INDICATOR,
        "Low MAPQ means ambiguous placement; alignment artefacts concentrate there.",
        min_value=0.0, max_value=60.0,
    ),
    Feature(
        "median_mapq", Group.BAM, "float",
        "Median mapping quality of reads overlapping the breakpoint.",
        Missing.NAN_INDICATOR,
        "Robust to a few high-MAPQ outliers that would skew the mean.",
        min_value=0.0, max_value=60.0,
    ),
    Feature(
        "frac_mapq0", Group.BAM, "float",
        "Fraction of overlapping reads with MAPQ equal to zero.",
        Missing.NAN_INDICATOR,
        "Annotation-free measure of repetitiveness; complements the Group A repeat tracks.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "n_split_reads", Group.BAM, "int",
        "Reads carrying an SA tag whose segments flank the breakpoint.",
        Missing.ZERO,
        "Split reads are the primary long-read signature of DEL, DUP, INV and BND.",
        min_value=0,
    ),
    Feature(
        "n_clipped_reads", Group.BAM, "int",
        "Reads soft- or hard-clipped by at least min_clip_length near the breakpoint.",
        Missing.ZERO,
        "Clipping clusters at real breakpoints; scattered clipping suggests noise.",
        min_value=0,
    ),
    Feature(
        "clip_frac", Group.BAM, "float",
        "n_clipped_reads divided by local depth.",
        Missing.NAN_INDICATOR,
        "Normalises the clip count for coverage so it is comparable across loci and samples.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "n_spanning_reads", Group.BAM, "int",
        "Reads crossing the whole span with no indel of at least min_sv_length in the CIGAR.",
        Missing.ZERO,
        "Reference-supporting evidence. Many spanning reads with little support suggests a FP.",
        min_value=0,
    ),
    Feature(
        "support_ratio", Group.BAM, "float",
        "Supporting reads divided by (supporting + clean spanning reads).",
        Missing.NAN_INDICATOR,
        "Allele-fraction proxy. Germline SVs cluster near 0.5 (het) or 1.0 (hom); "
        "values near 0.1 are typically artefacts.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "strand_balance", Group.BAM, "float",
        "Absolute strand asymmetry |fwd - rev| / (fwd + rev) among supporting reads.",
        Missing.NAN_INDICATOR,
        "Real germline SVs show balanced strands; strong bias indicates a systematic artefact.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "mean_read_len_local", Group.BAM, "float",
        "Mean aligned reference length of reads overlapping the breakpoint.",
        Missing.NAN_INDICATOR,
        "Short local reads reduce the chance of spanning a large SV; controls for data quality.",
        min_value=0.0,
    ),
    Feature(
        "mean_nm_rate", Group.BAM, "float",
        "Mean NM divided by aligned length over overlapping reads.",
        Missing.NAN_INDICATOR,
        "High local mismatch rate signals poor alignment, which generates spurious signatures.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "n_supplementary", Group.BAM, "int",
        "Count of supplementary alignments within the breakpoint window.",
        Missing.ZERO,
        "Excess supplementary alignments indicates multi-mapping or segmental duplication.",
        min_value=0,
    ),
]

# --------------------------------------------------------------------------------
# Group A -- annotation-derived
# --------------------------------------------------------------------------------
_GROUP_A = [
    Feature(
        "tr_overlap_frac", Group.ANNOT, "float",
        "Fraction of the SV span overlapping UCSC simpleRepeat.",
        Missing.ZERO,
        "Tandem repeats dominate long-read false-positive indels and also host many real SVs.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "rmsk_overlap_frac", Group.ANNOT, "float",
        "Fraction of the SV span overlapping UCSC RepeatMasker annotations.",
        Missing.ZERO,
        "Alu and LINE elements drive both real mobile-element insertions and mismapping.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "segdup_overlap", Group.ANNOT, "float",
        "Fraction of the SV span overlapping UCSC genomicSuperDups.",
        Missing.ZERO,
        "Segmental duplications cause paralogous mismapping, a top source of false DUP and INV.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "dist_to_nearest_gap", Group.ANNOT, "int",
        "Distance in bp from the breakpoint to the nearest assembly gap or N-run.",
        Missing.SENTINEL,
        "Calls adjacent to assembly gaps are frequently artefacts of the reference itself.",
        min_value=0,
    ),
    Feature(
        "gc_content", Group.ANNOT, "float",
        "GC fraction over the span plus flanking w_flank bases.",
        Missing.NAN_INDICATOR,
        "Extreme GC correlates with coverage bias and elevated basecalling error.",
        min_value=0.0, max_value=1.0,
    ),
    Feature(
        "homopolymer_len_at_bp", Group.ANNOT, "int",
        "Longest homopolymer run within 50 bp of the breakpoint.",
        Missing.NAN_INDICATOR,
        "Homopolymers are the classic ONT error mode; short indel artefacts cluster there.",
        min_value=1,
    ),
    Feature(
        "bp_microhomology_len", Group.ANNOT, "int",
        "Longest common sequence shared between the two breakpoint flanks.",
        Missing.NAN_INDICATOR,
        "Distinguishes repair-mechanism signatures of real SVs from alignment ambiguity.",
        min_value=0,
    ),
    Feature(
        "chrom_class", Group.ANNOT, "category",
        "Chromosome category: autosome, chrX, chrY or other.",
        Missing.ZERO,
        "Sex chromosomes have different expected ploidy, which changes how a depth ratio "
        "should be read. Raw chromosome identity is deliberately excluded -- it is the "
        "split grouping variable.",
        categorical=True,
    ),
]

# --------------------------------------------------------------------------------
# Group C -- caller-reported evidence (ablated)
# --------------------------------------------------------------------------------
_GROUP_C = [
    Feature(
        "sniffles_support", Group.CALLER, "float",
        "Sniffles2 supporting-read count (INFO/SUPPORT or equivalent alias).",
        Missing.NAN_INDICATOR,
        "The caller's own evidence count: strong, but not independent of the caller.",
        min_value=0,
    ),
    Feature(
        "cutesv_support", Group.CALLER, "float",
        "cuteSV supporting-read count (INFO/RE or equivalent alias).",
        Missing.NAN_INDICATOR,
        "Same signal from an independent implementation; disagreement between them is informative.",
        min_value=0,
    ),
    Feature(
        "n_callers", Group.CALLER, "int",
        "Number of callers supporting the merged candidate (1 or 2).",
        Missing.ZERO,
        "Encodes the intersection baseline, so the model can be shown to beat it rather "
        "than merely reproduce it.",
        min_value=1, max_value=2,
    ),
    Feature(
        "bp_offset_between_callers", Group.CALLER, "float",
        "Absolute difference between the two callers' reported breakpoint positions.",
        Missing.NAN_INDICATOR,
        "Tight agreement indicates a well-defined real breakpoint.",
        min_value=0,
    ),
    Feature(
        "svlen_disagreement", Group.CALLER, "float",
        "Absolute difference between the two callers' reported SV lengths.",
        Missing.NAN_INDICATOR,
        "Large disagreement flags an ambiguous or repeat-driven event.",
        min_value=0,
    ),
    Feature(
        "caller_qual", Group.CALLER, "float",
        "VCF QUAL, averaged when both callers report the candidate.",
        Missing.NAN_INDICATOR,
        "The callers' own confidence, kept for direct comparison against the learned score.",
        min_value=0,
    ),
    Feature(
        "filter_pass", Group.CALLER, "int",
        "1 if the candidate is PASS in at least one caller, else 0.",
        Missing.ZERO,
        "Encodes the callers' own built-in heuristic filter.",
        min_value=0, max_value=1,
    ),
]

# --------------------------------------------------------------------------------
# Group V -- caller-declared type and size (leakage-sensitive)
# --------------------------------------------------------------------------------
_GROUP_V = [
    Feature(
        "log10_svlen", Group.DECLARED, "float",
        "log10(|SVLEN| + 1).",
        Missing.NAN_INDICATOR,
        "SV length distribution is strongly type- and artefact-dependent "
        "(Alu near 300 bp, LINE near 6 kb).",
        min_value=0.0,
    ),
    Feature(
        "svtype_called", Group.DECLARED, "category",
        "Caller-declared SV type: DEL, INS, DUP, INV or BND.",
        Missing.ZERO,
        "LEAKAGE RISK for stage B, where it directly encodes the target and is excluded.",
        categorical=True,
    ),
    Feature(
        "span_len", Group.DECLARED, "int",
        "Reference interval length, END - POS. Zero for insertions.",
        Missing.ZERO,
        "Distinguishes reference-consuming events from insertions; pairs with dp_flank_ratio.",
        min_value=0,
    ),
]

FEATURES: list[Feature] = _GROUP_B + _GROUP_A + _GROUP_C + _GROUP_V
FEATURES_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}

# Identity columns carried through the matrix but never used as model inputs.
# `chrom` is the split grouping variable and `pos` is an arbitrary index; feeding
# either to the model would let it memorise loci (finding H3, leakage control 3).
ID_COLUMNS = ["candidate_id", "sample_id", "chrom", "pos", "end", "svtype", "svlen"]
LABEL_COLUMNS = ["is_true", "true_sv_type", "pseudo_sv_type", "label_status", "split"]
NON_FEATURE_COLUMNS = set(ID_COLUMNS) | set(LABEL_COLUMNS)

CATEGORICAL_FEATURES = [f.name for f in FEATURES if f.categorical]
CHROM_CLASSES = ["autosome", "chrX", "chrY", "other"]
SVTYPE_CLASSES = ["DEL", "INS", "DUP", "INV", "BND"]


def feature_names(groups: list[str] | None = None) -> list[str]:
    """Feature names, optionally restricted to the given ablation groups."""
    if groups is None:
        return [f.name for f in FEATURES]
    wanted = {Group(g) for g in groups}
    return [f.name for f in FEATURES if f.group in wanted]


def indicator_name(feature: str) -> str:
    return f"{feature}__isna"


def indicator_names(groups: list[str] | None = None) -> list[str]:
    return [
        indicator_name(f.name)
        for f in FEATURES
        if f.needs_indicator and (groups is None or f.group in {Group(g) for g in groups})
    ]


def model_columns(groups: list[str] | None = None, exclude: list[str] | None = None) -> list[str]:
    """Full model input column list for an ablation, after one-hot expansion."""
    exclude_set = set(exclude or [])
    cols: list[str] = []
    for feat in FEATURES:
        if groups is not None and feat.group not in {Group(g) for g in groups}:
            continue
        if feat.name in exclude_set:
            continue
        if feat.categorical:
            classes = CHROM_CLASSES if feat.name == "chrom_class" else SVTYPE_CLASSES
            cols.extend(f"{feat.name}__{c}" for c in classes)
        else:
            cols.append(feat.name)
            if feat.needs_indicator:
                cols.append(indicator_name(feat.name))
    return cols


def summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for feat in FEATURES:
        counts[feat.group.value] = counts.get(feat.group.value, 0) + 1
    counts["total"] = len(FEATURES)
    return counts


def to_markdown() -> str:
    """Render the registry as the feature dictionary table (spec section 27)."""
    lines = [
        "| # | Feature | Group | Type | Definition | Range | Missing | Biological meaning |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, f in enumerate(FEATURES, start=1):
        lo = "-" if f.min_value is None else f"{f.min_value:g}"
        hi = "-" if f.max_value is None else f"{f.max_value:g}"
        rng = "categorical" if f.categorical else f"{lo}..{hi}"
        lines.append(
            f"| {i} | `{f.name}` | {f.group.value} | {f.dtype} | {f.definition} "
            f"| {rng} | {f.missing.value} | {f.meaning} |"
        )
    return "\n".join(lines)
