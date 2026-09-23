"""Stage 9 -- benchmarking against all six baselines (finding H2).

This is where the test split is read, exactly once, with models and thresholds already
frozen by stage 8.

Stage C is deliberately absent: it is pseudo-labelled and must never be benchmarked
as ground truth (finding B1). `BENCHMARK_STAGES` excludes it via config.
"""


def benchmark_models(wildcards):
    """The full-feature models for a stage. The ablation models are reported separately.

    Uses models_for_stage(), not the flat MODELS list: Stage C excludes XGBoost, which
    cannot train on its sparse pseudo-label distribution at pilot scale (D16/D17).
    """
    full = ABLATIONS[-1]
    return [
        str(RESULTS / "models" / f"stage{wildcards.stage}_{model}_{full}.pkl")
        for model in models_for_stage(wildcards.stage)
    ]


rule benchmark_stage:
    input:
        dataset=RESULTS / "ml" / "dataset.parquet",
        models=benchmark_models,
        split_check=RESULTS / "checks" / "split_check.txt",
        gate=RESULTS / "labels" / f"{SAMPLE}.gate_summary.json",
    output:
        metrics=RESULTS / "benchmark" / "stage{stage}_metrics.json",
    log:
        LOGS / "benchmark" / "stage{stage}.log",
    shell:
        r"""
        mkdir -p $(dirname {output.metrics}) $(dirname {log})
        svml benchmark --config config/config.yaml \
          --dataset {input.dataset} --stage {wildcards.stage} \
          --models {input.models} --gate-summary {input.gate} \
          --out {output.metrics} --log {log}
        """


rule benchmark_ablation:
    """The B+A model: does independent evidence add anything beyond the caller fields?"""
    input:
        dataset=RESULTS / "ml" / "dataset.parquet",
        models=expand(
            str(RESULTS / "models" / "stage{{stage}}_{model}_{ablation}.pkl"),
            model=MODELS,
            ablation=[ABLATIONS[0]],
        ),
        gate=RESULTS / "labels" / f"{SAMPLE}.gate_summary.json",
    output:
        metrics=RESULTS / "benchmark" / "stage{stage}_ablation_metrics.json",
    log:
        LOGS / "benchmark" / "stage{stage}_ablation.log",
    shell:
        r"""
        svml benchmark --config config/config.yaml \
          --dataset {input.dataset} --stage {wildcards.stage} \
          --models {input.models} --gate-summary {input.gate} \
          --out {output.metrics} --log {log}
        """


rule error_analysis:
    """Required, not optional (docs/05_evaluation_protocol.md).

    Emits FP/FN samples for IGV curation. Some apparent errors in a draft benchmark are
    representation differences or benchmark errors rather than model errors, and the
    report must distinguish them.
    """
    input:
        dataset=RESULTS / "ml" / "dataset.parquet",
        metrics=RESULTS / "benchmark" / "stage{stage}_metrics.json",
        model=RESULTS / "models" / f"stage{{stage}}_random_forest_{ABLATIONS[-1]}.pkl",
    output:
        fp=RESULTS / "benchmark" / "stage{stage}_false_positives.tsv",
        fn=RESULTS / "benchmark" / "stage{stage}_false_negatives.tsv",
        summary=RESULTS / "benchmark" / "stage{stage}_error_analysis.json",
    params:
        n=50,
    log:
        LOGS / "benchmark" / "stage{stage}_errors.log",
    shell:
        r"""
        python - <<'PY' 2> {log}
import json
import numpy as np
import pandas as pd
from svml.utils.config import load_config
from svml.ml.splits import assign_splits
from svml.ml.train import load_model, predict
from svml.ml import models as model_defs

cfg = load_config("config/config.yaml")
stage_cfg = cfg.require("ml.stages.{wildcards.stage}")
target = stage_cfg["target"]

frame = pd.read_parquet("{input.dataset}")
frame = assign_splits(frame, cfg.split_map())
test = frame[(frame["split"] == "test") & frame[target].notna()].copy()

model, encoder, meta = load_model("{input.model}")
probs = predict(model, encoder, test, meta.model_name)
scores = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
threshold = meta.threshold if meta.threshold is not None else 0.5

test["svml_score"] = scores
test["svml_pred"] = (scores >= threshold).astype(int)
y = test[target].astype(int)

cols = ["candidate_id", "chrom", "pos", "end", "svtype", "svlen", "svml_score",
        "n_callers", "tr_overlap_frac", "segdup_overlap", "frac_mapq0",
        "dp_flank_ratio", "support_ratio"]
cols = [c for c in cols if c in test.columns]

fp = test[(y == 0) & (test["svml_pred"] == 1)].nlargest({params.n}, "svml_score")
fn = test[(y == 1) & (test["svml_pred"] == 0)].nsmallest({params.n}, "svml_score")

fp[cols].to_csv("{output.fp}", sep="\t", index=False)
fn[cols].to_csv("{output.fn}", sep="\t", index=False)

def strat(sub):
    out = {{}}
    if "tr_overlap_frac" in sub.columns:
        out["in_tandem_repeat"] = int((sub["tr_overlap_frac"] > 0.5).sum())
    if "segdup_overlap" in sub.columns:
        out["in_segdup"] = int((sub["segdup_overlap"] > 0).sum())
    if "n_callers" in sub.columns:
        out["single_caller"] = int((sub["n_callers"] < 2).sum())
    return out

summary = {{
    "false_positives": {{
        "count": int(((y == 0) & (test["svml_pred"] == 1)).sum()),
        "sampled": int(len(fp)),
        "strata": strat(fp),
        "note": "Curate in IGV. Distinguish genuine model errors from representation "
                "differences and benchmark errors (v5.0q is a draft benchmark).",
    }},
    "false_negatives": {{
        "count": int(((y == 1) & (test["svml_pred"] == 0)).sum()),
        "sampled": int(len(fn)),
        "strata": strat(fn),
        "note": "Curate in IGV as above.",
    }},
    "curation_status": {{
        "count": 0,
        "note": "Manual IGV curation not yet performed. Required before the report is "
                "final (docs/05_evaluation_protocol.md, error analysis step 4).",
    }},
}}
json.dump(summary, open("{output.summary}", "w"), indent=2)
print(json.dumps(summary, indent=2))
PY
        """


rule final_vcf:
    """Output I: the final annotated VCF, with scores attached and nothing discarded."""
    input:
        candidate_vcf=RESULTS / "candidates" / "{sample}.merged.vcf.gz",
        dataset=RESULTS / "ml" / "dataset.parquet",
        model=RESULTS / "models" / f"stageA_random_forest_{ABLATIONS[-1]}.pkl",
    output:
        vcf=RESULTS / "final" / "{sample}.svml.vcf.gz",
    log:
        LOGS / "benchmark" / "{sample}.final_vcf.log",
    shell:
        r"""
        mkdir -p $(dirname {output.vcf}) $(dirname {log})
        python - <<'PY' 2> {log}
import pandas as pd
from svml.utils.config import load_config
from svml.io.vcf import write_annotated_vcf
from svml.ml.train import load_model, predict

cfg = load_config("config/config.yaml")
frame = pd.read_parquet("{input.dataset}")
frame = frame[frame["sample_id"] == "{wildcards.sample}"]

model, encoder, meta = load_model("{input.model}")
probs = predict(model, encoder, frame, meta.model_name)
scores = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
threshold = meta.threshold if meta.threshold is not None else 0.5

predictions = {{
    str(row.candidate_id): {{
        "score": float(score),
        "predicted_type": row.svtype,
        "label_status": getattr(row, "label_status", None),
    }}
    for row, score in zip(frame.itertuples(index=False), scores)
}}

write_annotated_vcf("{input.candidate_vcf}", "{output.vcf}", predictions,
                    filter_threshold=threshold)
print(f"annotated {{len(predictions)}} candidates at threshold {{threshold:.3f}}")
PY
        """
