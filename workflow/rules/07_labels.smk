"""Stage 7 -- labelling with the confident-region gate (finding H4).

Only labelled-role samples reach this stage; the rest carry no truth and are used for
candidate generation and robustness demonstration only.
"""


rule assign_labels:
    input:
        features=RESULTS / "features" / "{sample}.parquet",
        candidate_vcf=RESULTS / "candidates" / "{sample}.merged.vcf.gz",
        candidate_tbi=RESULTS / "candidates" / "{sample}.merged.vcf.gz.tbi",
        truth=TRUTH_VCF,
        confident=CONFIDENT_BED,
        fa=REFERENCE_FA,
        sanity=RESULTS / "checks" / "{sample}.feature_sanity.json",
    output:
        parquet=RESULTS / "labels" / "{sample}.labelled.parquet",
        summary=RESULTS / "labels" / "{sample}.gate_summary.json",
        bench=directory(RESULTS / "labels" / "{sample}.truvari"),
    log:
        LOGS / "labels" / "{sample}.label.log",
    shell:
        r"""
        mkdir -p $(dirname {output.parquet}) $(dirname {log})
        svml label --config config/config.yaml \
          --features {input.features} --candidate-vcf {input.candidate_vcf} \
          --truth-vcf {input.truth} --confident-bed {input.confident} \
          --reference {input.fa} --bench-dir {output.bench} \
          --out {output.parquet} --out-summary {output.summary} --log {log}
        """


rule label_check:
    """Phase 8 exit test.

    A very large unlabelled fraction means the pilot region is poorly covered by
    confident regions -- reconsider the region rather than training on a handful of
    labels.
    """
    input:
        summary=RESULTS / "labels" / "{sample}.gate_summary.json",
    output:
        ok=RESULTS / "checks" / "{sample}.label_check.txt",
    log:
        LOGS / "labels" / "{sample}.check.log",
    shell:
        r"""
        mkdir -p $(dirname {output.ok})
        python - <<'PY' > {output.ok} 2> {log}
import json, sys

counts = json.load(open("{input.summary}"))
total = counts.get("total_candidates", 0)
inside = counts.get("inside_confident", 0)
n_true = counts.get("labelled_true", 0)
n_false = counts.get("labelled_false", 0)

print(json.dumps(counts, indent=2))
print()
if total:
    print(f"inside confident regions : {{inside}}/{{total}} ({{100*inside/total:.1f}}%)")
print(f"labelled true  : {{n_true}}")
print(f"labelled false : {{n_false}}")

if n_true == 0 or n_false == 0:
    print("LABEL CHECK FAILED: one class is empty; a classifier cannot be trained.",
          file=sys.stderr)
    raise SystemExit(1)
if n_true + n_false < 200:
    print(f"WARNING: only {{n_true + n_false}} labelled candidates. Consider a larger "
          "region before training (Phase 8 exit test).", file=sys.stderr)
print("\nlabel check passed")
PY
        """


rule build_dataset:
    """Combine labelled samples into the modelling dataset."""
    input:
        parquets=expand(str(RESULTS / "labels" / "{sample}.labelled.parquet"), sample=LABELLED),
        checks=expand(str(RESULTS / "checks" / "{sample}.label_check.txt"), sample=LABELLED),
    output:
        parquet=RESULTS / "ml" / "dataset.parquet",
    log:
        LOGS / "labels" / "build_dataset.log",
    shell:
        r"""
        mkdir -p $(dirname {output.parquet})
        python - <<'PY' 2> {log}
import pandas as pd

frames = [pd.read_parquet(p) for p in "{input.parquets}".split()]
combined = pd.concat(frames, ignore_index=True)
combined.to_parquet("{output.parquet}", index=False)
print(f"dataset: {{len(combined)}} rows from {{len(frames)}} labelled sample(s)")
PY
        """
