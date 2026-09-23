"""Stage 11 -- reporting (spec section 19: all 20 sections; outputs P and Q).

Missing sections render as TBD. Nothing illustrative ever enters a results table
(spec section 28).
"""


rule run_metadata:
    """Provenance: tool versions, seeds, dataset versions, config hash, date (section 23)."""
    output:
        json=RESULTS / "report" / "run_metadata.json",
    log:
        LOGS / "report" / "metadata.log",
    shell:
        r"""
        mkdir -p $(dirname {output.json}) $(dirname {log})
        python - <<'PY' 2> {log}
from svml.utils.config import load_config
from svml.utils.provenance import write_metadata

cfg = load_config("config/config.yaml")
write_metadata(cfg, "{output.json}")
PY
        """


rule build_report:
    input:
        dataset=RESULTS / "ml" / "dataset.parquet",
        benchmark=RESULTS / "benchmark" / "stageA_metrics.json",
        errors=RESULTS / "benchmark" / "stageA_error_analysis.json",
        gate=expand(str(RESULTS / "labels" / "{sample}.gate_summary.json"), sample=[SAMPLE]),
        candidates=expand(
            str(RESULTS / "candidates" / "{sample}.merge_summary.json"), sample=[SAMPLE]
        ),
        qc=expand(str(RESULTS / "qc" / "{sample}" / "qc_metrics.json"), sample=ALL_SAMPLES),
        align=expand(str(RESULTS / "align" / "{sample}.align_stats.json"), sample=ALL_SAMPLES),
        shap=expand(
            str(RESULTS / "shap" / "shap_summary_{stage}_{model}_{ablation}.json"),
            stage=["A"],
            model=MODELS,
            ablation=[ABLATIONS[-1]],
        ),
        metadata=RESULTS / "report" / "run_metadata.json",
        splits=RESULTS / "ml" / f"splits_stageA_random_forest_{ABLATIONS[-1]}.json",
        ablation=expand(
            str(RESULTS / "benchmark" / "stage{stage}_ablation_metrics.json"),
            stage=BENCHMARK_STAGES,
        ),
        stage_c=RESULTS / "benchmark" / "stageC_metrics.json",
    output:
        html=RESULTS / "report" / "report.html",
        json=RESULTS / "report" / "summary.json",
        pdf=RESULTS / "report" / "report.pdf",
    log:
        LOGS / "report" / "build.log",
    shell:
        r"""
        mkdir -p $(dirname {output.html}) $(dirname {log})
        svml report --config config/config.yaml \
          --dataset {input.dataset} \
          --benchmark {input.benchmark} \
          --gate-summary {input.gate[0]} \
          --candidate-summary {input.candidates[0]} \
          --qc {input.qc} \
          --align-stats {input.align} \
          --splits {input.splits} \
          --errors {input.errors} \
          --ablation-metrics {input.ablation} \
          --stage-c {input.stage_c} \
          --shap {input.shap} \
          --out-html {output.html} \
          --out-json {output.json} \
          --out-pdf {output.pdf} \
          --log {log}
        touch {output.pdf}
        """


rule feature_dictionary:
    """Regenerate the feature dictionary table from the registry, so docs cannot drift."""
    output:
        md=RESULTS / "report" / "feature_dictionary_generated.md",
    log:
        LOGS / "report" / "feature_docs.log",
    shell:
        "svml feature-docs --config config/config.yaml --out {output.md} --log {log}"
