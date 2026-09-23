"""svml -- AI-based structural variant detection using long-read sequencing data.

    snakemake --cores 4 --use-conda

Must run inside WSL2 or Docker: minimap2, samtools, sniffles, cuteSV, bcftools,
SURVIVOR and truvari have no native Windows builds (finding M4).

Stage order follows docs/01_system_architecture.md:
    0 references -> 1 acquire -> 2 qc -> 3 align -> 4 call -> 5 merge
      -> 6 features -> 7 labels -> 8 train -> 9 benchmark -> 10 explain -> 11 report
"""

import os

import pandas as pd


class P(str):
    """A path that is a `str` and always joins with POSIX separators.

    The pipeline runs under WSL/Linux (finding M4). Using pathlib on a Windows host
    produces backslashes, which breaks Snakemake's input/output matching between rules
    during a dry run, while a plain PurePosixPath is rejected as a rule output.
    Subclassing `str` satisfies Snakemake and keeps the `DIR / "name"` rule syntax.
    """

    __slots__ = ()

    def __truediv__(self, other: object) -> "P":
        return P(f"{self.rstrip('/')}/{str(other).lstrip('/')}")

configfile: "config/config.yaml"


# --- manifests --------------------------------------------------------------
SAMPLES = pd.read_csv(config["samples"]["manifest"], sep="\t", comment="#", dtype=str).fillna("")
SAMPLES = SAMPLES.set_index("sample_id", drop=False)

import yaml
with open(config["reference"]["manifest"]) as _fh:
    REFS = yaml.safe_load(_fh)


# --- paths ------------------------------------------------------------------
WORK = P(os.path.expanduser(config["project"]["workdir"]).replace("\\", "/"))
RESOURCES = WORK / "resources"
RESULTS = WORK / "results"
LOGS = RESULTS / "logs"

REFERENCE_FA = RESOURCES / "reference" / "GRCh38.fa"
REFERENCE_MMI = RESOURCES / "reference" / "GRCh38.mmi"
ANNOTATION_DIR = RESOURCES / "annotations"
TRUTH_VCF = RESOURCES / "truth" / "truth.vcf.gz"
CONFIDENT_BED = RESOURCES / "truth" / "confident_regions.bed"


# --- wildcard sets ----------------------------------------------------------
ALL_SAMPLES = SAMPLES["sample_id"].tolist()
LABELLED = SAMPLES.loc[SAMPLES["role"] == "labelled", "sample_id"].tolist()
MODELS = ["random_forest", "xgboost"]
ABLATIONS = [a["name"] for a in config["ml"]["ablations"]]
STAGES = [s for s in ("A", "B", "C") if config["ml"]["stages"][s].get("enabled", True)]

# Stage A is the primary benchmarked experiment; stage C is pseudo-labelled, so it is
# trained and reported descriptively but never benchmarked as ground truth (finding B1).
BENCHMARK_STAGES = [s for s in STAGES if config["ml"]["stages"][s].get("benchmarked", True)]
# Stage C still gets a metrics file -- it just carries a label_provenance field marking
# the numbers as pseudo-label derived.
EVALUATED_STAGES = STAGES


def models_for_stage(stage):
    """Which models to require for a given stage (finding D16/D17).

    Stage C's five pseudo-label classes are severely imbalanced at Tier-1 pilot scale
    (DUP and TRA can be entirely absent from the training fold). XGBoost's sklearn
    wrapper requires contiguous class indices in y_train and raises rather than
    degrading; Random Forest trains successfully on the same data. Requiring XGBoost
    for Stage C would make the DAG permanently unsatisfiable on this pilot's data,
    not something a retry fixes. This is a data-scale limitation, not a code defect --
    Stage A and B, both fully benchmarked, are unaffected and still require both models.
    """
    if stage == "C":
        return ["random_forest"]
    return MODELS

def regions_for_role(role):
    """Region strings to acquire for a sample role (finding M2)."""
    if int(config["regions"]["tier"]) == 1:
        tier1 = config["regions"]["tier1"]
        if isinstance(tier1, list):
            return tier1
        return tier1.get(role, tier1["labelled"])
    tier2 = config["regions"]["tier2"]
    return tier2["train"] + tier2["val"] + tier2["test"]


def sample_regions(wildcards):
    return " ".join(regions_for_role(SAMPLES.loc[wildcards.sample, "role"]))


def sample_preset(wildcards):
    return SAMPLES.loc[wildcards.sample, "preset"]


def primary_labelled():
    if not LABELLED:
        raise ValueError("no sample has role='labelled'; cannot train a model")
    return LABELLED[0]


SAMPLE = primary_labelled()


# Ablation names share prefixes ("B+A" is a prefix of "B+A+C+V"), and regex alternation
# is first-match rather than longest-match, so the longer names must come first or
# "stageA_rf_B+A+C+V.pkl" would resolve with ablation="B+A".
_ABLATION_PATTERN = "|".join(
    a.replace("+", r"\+") for a in sorted(ABLATIONS, key=len, reverse=True)
)


wildcard_constraints:
    sample="|".join(ALL_SAMPLES) if ALL_SAMPLES else "NONE",
    model="|".join(MODELS),
    stage="[ABC]",
    ablation=_ABLATION_PATTERN,


include: "workflow/rules/00_references.smk"
include: "workflow/rules/01_acquire.smk"
include: "workflow/rules/02_qc.smk"
include: "workflow/rules/03_align.smk"
include: "workflow/rules/04_call.smk"
include: "workflow/rules/05_merge.smk"
include: "workflow/rules/06_features.smk"
include: "workflow/rules/07_labels.smk"
include: "workflow/rules/08_train.smk"
include: "workflow/rules/09_benchmark.smk"
include: "workflow/rules/10_explain.smk"
include: "workflow/rules/11_report.smk"


rule all:
    input:
        # Every stage, model and ablation -- the ablation (finding M3) is produced by
        # construction rather than as an afterthought, and stage C still ships as a
        # deliverable even though it is never benchmarked (finding B1). Models are
        # requested per-stage (models_for_stage) since Stage C excludes XGBoost, which
        # cannot train on its sparse pseudo-label distribution at pilot scale (D16/D17).
        [
            str(RESULTS / "models" / f"stage{stage}_{model}_{ablation}.pkl")
            for stage in STAGES
            for model in models_for_stage(stage)
            for ablation in ABLATIONS
        ],
        expand(
            str(RESULTS / "benchmark" / "stage{stage}_metrics.json"), stage=EVALUATED_STAGES
        ),
        expand(
            str(RESULTS / "benchmark" / "stage{stage}_ablation_metrics.json"),
            stage=BENCHMARK_STAGES,
        ),
        RESULTS / "report" / "report.html",
        RESULTS / "report" / "summary.json",
        RESULTS / "report" / "run_metadata.json",
        RESULTS / "report" / "feature_dictionary_generated.md",
        RESULTS / "final" / f"{SAMPLE}.svml.vcf.gz",


rule milestone1:
    """FIRST MILESTONE (spec section 34): raw reads through to validated candidate VCFs."""
    input:
        expand(str(RESULTS / "qc" / "{sample}" / "qc_metrics.json"), sample=ALL_SAMPLES),
        expand(str(RESULTS / "align" / "{sample}.sorted.bam.bai"), sample=ALL_SAMPLES),
        expand(str(RESULTS / "sv" / "{sample}.sniffles.vcf.gz"), sample=ALL_SAMPLES),
        expand(str(RESULTS / "sv" / "{sample}.cutesv.vcf.gz"), sample=ALL_SAMPLES),
        expand(str(RESULTS / "candidates" / "{sample}.merge_summary.json"), sample=ALL_SAMPLES),


rule milestone2:
    """SECOND MILESTONE: candidate VCFs through to the feature matrix."""
    input:
        expand(str(RESULTS / "features" / "{sample}.parquet"), sample=ALL_SAMPLES),


rule milestone3:
    """THIRD MILESTONE: feature matrix through to trained models and evaluation."""
    input:
        expand(
            str(RESULTS / "benchmark" / "stage{stage}_metrics.json"),
            stage=BENCHMARK_STAGES,
        ),


rule check_env:
    """Phase 0 exit test."""
    output:
        touch(RESULTS / "checks" / "env.ok"),
    shell:
        "svml check-env --config config/config.yaml"
