"""Stage 10 -- SHAP explainability (spec section 17).

SHAP is an explanation layer: it does not alter predictions and does not replace the
classifier.
"""


rule shap_explain:
    input:
        model=RESULTS / "models" / "stage{stage}_{model}_{ablation}.pkl",
        dataset=RESULTS / "ml" / "dataset.parquet",
    output:
        summary=RESULTS / "shap" / "shap_summary_{stage}_{model}_{ablation}.json",
        out_dir=directory(RESULTS / "shap" / "stage{stage}_{model}_{ablation}"),
    log:
        LOGS / "explain" / "stage{stage}_{model}_{ablation}.log",
    shell:
        r"""
        mkdir -p {output.out_dir} $(dirname {log})
        svml shap --config config/config.yaml \
          --model {input.model} --dataset {input.dataset} \
          --out-dir {output.out_dir} --log {log}
        cp {output.out_dir}/shap_summary_*.json {output.summary}
        """


rule all_shap:
    input:
        expand(
            str(RESULTS / "shap" / "shap_summary_{stage}_{model}_{ablation}.json"),
            stage=BENCHMARK_STAGES,
            model=MODELS,
            ablation=[ABLATIONS[-1]],
        ),
