"""Stage 8 -- training.

Every (stage, model, ablation) combination is trained, so the required ablation
(finding M3) is produced by construction rather than as an afterthought.

The test split is never read by these rules. Hyperparameter search and threshold
selection use validation only.
"""


rule train_model:
    input:
        dataset=RESULTS / "ml" / "dataset.parquet",
    output:
        model=RESULTS / "models" / "stage{stage}_{model}_{ablation}.pkl",
        meta=RESULTS / "models" / "stage{stage}_{model}_{ablation}.json",
        splits=RESULTS / "ml" / "splits_stage{stage}_{model}_{ablation}.json",
    threads: config["align"]["threads"]
    log:
        LOGS / "train" / "stage{stage}_{model}_{ablation}.log",
    shell:
        r"""
        mkdir -p $(dirname {output.model}) $(dirname {log})
        svml train --config config/config.yaml \
          --dataset {input.dataset} --stage {wildcards.stage} \
          --model {wildcards.model} --ablation "{wildcards.ablation}" \
          --out-model {output.model} --out-meta {output.meta} \
          --out-splits {output.splits} --log {log}
        """


rule all_models:
    input:
        expand(
            str(RESULTS / "models" / "stage{stage}_{model}_{ablation}.pkl"),
            stage=STAGES,
            model=MODELS,
            ablation=ABLATIONS,
        ),


rule split_check:
    """Asserts the leakage invariant: no chromosome appears in more than one split."""
    input:
        dataset=RESULTS / "ml" / "dataset.parquet",
    output:
        ok=RESULTS / "checks" / "split_check.txt",
    log:
        LOGS / "train" / "split_check.log",
    shell:
        r"""
        mkdir -p $(dirname {output.ok})
        python - <<'PY' > {output.ok} 2> {log}
import json
import pandas as pd
from svml.utils.config import load_config
from svml.ml.splits import assign_splits, class_distribution, validate_splits

cfg = load_config("config/config.yaml")
frame = pd.read_parquet("{input.dataset}")
frame = assign_splits(frame, cfg.split_map())
validate_splits(frame)

print("split sizes:")
print(frame["split"].value_counts(dropna=False).to_string())
print()
print("class distribution (is_true):")
print(json.dumps(class_distribution(frame, "is_true"), indent=2))
print()
print("split check passed: no chromosome spans two splits")
PY
        """
