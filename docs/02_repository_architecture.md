# Repository Architecture

Fills PROJECT MASTER CONTEXT §20, which was left blank.

Principle: **config declares, rules orchestrate, modules compute, results are disposable.**
Everything under `results/` and `resources/` must be reconstructible from `config/` + code.

```
dna-project/
├── README.md                     reproduction instructions (§23)
├── environment.yml               pinned conda/mamba environment
├── Snakefile                     top-level orchestrator
├── .gitignore                    excludes resources/ and results/
│
├── config/
│   ├── config.yaml               run parameters, tool flags, thresholds, seeds
│   ├── samples.tsv               sample manifest: id, platform, coverage, source
│   └── references.yaml           genome + annotation + truth assets (url, md5, build)
│
├── workflow/
│   ├── rules/
│   │   ├── 00_references.smk     genome fetch, faidx, minimap2 index, annotation BEDs
│   │   ├── 01_acquire.smk        region slicing, BAM -> FASTQ
│   │   ├── 02_qc.smk             NanoPlot, metrics parsing, accept/flag
│   │   ├── 03_align.smk          minimap2, sort, index, samtools stats
│   │   ├── 04_call.smk           Sniffles2, cuteSV
│   │   ├── 05_merge.smk          bcftools norm, SURVIVOR/Jasmine merge
│   │   ├── 06_features.smk       feature extraction
│   │   ├── 07_labels.smk         confident-region gate, truvari matching
│   │   ├── 08_train.smk          RF / XGBoost, all stages, all ablations
│   │   ├── 09_benchmark.smk      truvari bench vs 6 baselines
│   │   ├── 10_explain.smk        SHAP
│   │   └── 11_report.smk         plots, HTML/PDF, summary JSON
│   ├── envs/                     per-rule conda envs (isolates tool conflicts)
│   │   ├── align.yaml
│   │   ├── sv.yaml
│   │   ├── bench.yaml
│   │   └── ml.yaml
│   └── schemas/                  JSON schemas validating config + samples.tsv
│
├── src/svml/                     importable, unit-testable python package
│   ├── __init__.py
│   ├── io/
│   │   ├── vcf.py                VCF read/write, INFO field extraction
│   │   ├── bam.py                pysam wrappers, depth and read queries
│   │   └── bed.py                interval overlap utilities
│   ├── features/
│   │   ├── registry.py           feature registry; single source of truth for schema
│   │   ├── bam_features.py       Group B
│   │   ├── annot_features.py     Group A
│   │   └── caller_features.py    Groups C and V
│   ├── labels/
│   │   ├── gate.py               confident-region gate (H4)
│   │   └── match.py              truvari result -> is_true / true_type
│   ├── ml/
│   │   ├── dataset.py            load, impute, encode
│   │   ├── splits.py             chromosome-grouped splitting (H3)
│   │   ├── models.py             RF / XGBoost definitions
│   │   ├── train.py              training + hyperparameter search
│   │   └── calibrate.py          probability calibration
│   ├── eval/
│   │   ├── baselines.py          union / intersection / per-caller baselines (H2)
│   │   ├── metrics.py            P / R / F1 / macro-F1 / PR curves
│   │   └── strata.py             repeat-context stratification (L1)
│   ├── explain/shap_runner.py
│   ├── report/
│   │   ├── plots.py              histograms, karyotype, confusion matrices
│   │   └── build.py              HTML/PDF assembly
│   └── utils/
│       ├── config.py             config loading + validation
│       ├── logging.py            structured logging (§36)
│       └── provenance.py         tool versions, seeds, timestamps (§23)
│
├── resources/                    NOT in git — fetched by stage 0/1
│   ├── reference/                GRCh38 fa, fai, mmi
│   ├── annotations/              rmsk, simpleRepeat, segdups, gaps
│   ├── truth/                    GIAB v5.0q stvar VCF + confident BED
│   └── raw/                      region-sliced FASTQ
│
├── results/                      NOT in git — fully reproducible
│   ├── qc/  align/  sv/  candidates/  features/  labels/
│   ├── ml/          splits.json, dataset.parquet
│   ├── models/      rf_*.pkl, xgb_*.pkl, model_metadata.json
│   ├── predictions/
│   ├── benchmark/   metrics.json, per-baseline truvari output
│   ├── shap/
│   └── report/      report.html, summary.json, run_metadata.json
│
├── notebooks/                    exploration only; never a pipeline dependency
│   ├── 01_candidate_inspection.ipynb
│   ├── 02_feature_sanity.ipynb
│   └── 03_error_analysis.ipynb
│
├── tests/                        pytest (§22)
│   ├── data/                     tiny fixtures: mini BAM, mini VCF, mini BED
│   ├── test_features.py          each feature on a known fixture
│   ├── test_gate.py              the three-way label outcome (H4)
│   ├── test_splits.py            asserts no chromosome spans two splits
│   └── test_metrics.py
│
└── docs/                         this directory — Phase 1 deliverables
    ├── 00_architecture_evaluation.md
    ├── 01_system_architecture.md
    ├── 02_repository_architecture.md
    ├── 03_feature_dictionary.md
    ├── 04_ml_task_definition.md
    ├── 05_evaluation_protocol.md
    └── 06_execution_plan.md
```

## Conventions

- **No path is hardcoded.** Every input/output path derives from `config.yaml` (§21).
- **`src/svml` is importable and tested independently of Snakemake.** Rules are thin
  wrappers that call into it. This is what makes `tests/` meaningful.
- **`features/registry.py` is the single source of truth for the feature schema.** The
  feature dictionary doc and the extraction code both read from it, so they cannot drift.
- **Every rule writes a log** to `results/logs/{rule}/{wildcards}.log` (§36).
- **`provenance.py` runs at the end of every pipeline invocation** and writes tool
  versions, config hash, seeds, dataset versions, and execution date (§23).
- **`resources/` and `results/` are git-ignored.** The repo holds code and config only, so
  a clean clone plus `snakemake --cores 4` reproduces everything.
