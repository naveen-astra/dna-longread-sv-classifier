# AI-Based Structural Variant Detection Using Long-Read Sequencing Data

Course: DNA Sequencing Technologies (22AIE455)
Type: Bioinformatics + Machine Learning + Explainable AI

A reproducible pipeline that aligns human long-read sequencing data to GRCh38, generates
candidate structural variants with Sniffles2 and cuteSV, extracts structured evidence
features from those candidates, trains Random Forest and XGBoost classifiers, benchmarks
against the GIAB HG002 truth set and against the heuristic callers, and produces an
interpretable report.

The ML layer **does not replace** Minimap2, Sniffles2 or cuteSV. It scores and filters the
candidates they produce.

---

## Status

| | |
|---|---|
| Architecture (Phase 1) | **Complete** — [docs/](docs/) |
| Pipeline implementation | **Complete** — Snakemake workflow + `svml` package, 76 unit tests passing |
| Pipeline **execution** | **Not run.** Requires WSL2, the conda toolchain, and GIAB downloads |
| Results | **None.** No metric in this repo is a measured result — placeholders render as `TBD` |

Every stage is coded and unit-tested, but the pipeline has not been executed against real
data, so the repository contains **no experimental results and no benchmark numbers**.
Per spec §28, none have been invented to fill the gap. Three config fields are
deliberately left as `TODO` (truth filenames, sample source URLs) and the pipeline fails
loudly on them rather than guessing — resolve them in Phase 2/3, then run.

---

## Read this first

The architecture review found one **blocking** issue that changes what the project can
claim. Full detail in [docs/00_architecture_evaluation.md](docs/00_architecture_evaluation.md).

**No GIAB truth set labels DUP, INV or TRA.** Verified against the live GIAB release tree:

- `NIST_SV_v0.6` — GRCh37 only; its README states *"All now are DEL or INS"*.
- `v5.0q` `stvar` — GRCh38-native, but assembly-derived and its benchmark regions exclude
  inversion interiors.
- `NISTv4.2.1` — small variants (SNV/indel), **not** SVs. The spec cited this as the SV
  ground truth; that is an error.

So objective O3 — a five-class classifier benchmarked against GIAB at macro-F1 ≥ 0.85 —
cannot be built as written. The labels do not exist. The resolution, per §13 Option C:

| Stage | Task | Labels | Benchmarked? |
|---|---|---|---|
| **A** | Candidate validity (true/false) | GIAB v5.0q | **Yes — primary** |
| **B** | SV type, DEL vs INS | GIAB v5.0q | **Yes — secondary** |
| **C** | Five-class INS/DEL/DUP/INV/TRA | Caller-consensus pseudo-labels | **No — exploratory** |

Stage A is also the task that actually answers the research question, which is about
*precision*, not type. The five-class deliverable still ships, labelled honestly.

### Three decisions need sign-off before Phase 2

1. **O3 restated** — macro-F1 ≥ 0.85 now applies to Stage A and Stage B. If the course
   requires a *benchmarked* five-class result, raise it with the instructor now.
2. **Truth set** switched to GIAB `v5.0q` `stvar` GRCh38, preserving the mandated GRCh38.
3. **Scope** reduced to selected chromosomes rather than whole genome, per §24.

---

## Documentation

| Document | Contents |
|---|---|
| [00_architecture_evaluation.md](docs/00_architecture_evaluation.md) | Review of the proposed architecture; 13 findings with fixes |
| [01_system_architecture.md](docs/01_system_architecture.md) | Revised system, data-flow and ML-flow diagrams |
| [02_repository_architecture.md](docs/02_repository_architecture.md) | Directory layout and conventions |
| [03_feature_dictionary.md](docs/03_feature_dictionary.md) | 33 draft features across 4 ablation groups |
| [04_ml_task_definition.md](docs/04_ml_task_definition.md) | Task choice, labelling rules, splits, leakage control |
| [05_evaluation_protocol.md](docs/05_evaluation_protocol.md) | Six baselines, PR curves, stratification, error analysis |
| [06_execution_plan.md](docs/06_execution_plan.md) | Phases 0–13 with an exit test for each |

---

## Repository layout

```
Snakefile                 orchestrator; `all` + milestone1/2/3 targets
workflow/rules/*.smk      11 stages, 00_references → 11_report
workflow/envs/*.yaml      per-rule conda environments
config/                   config.yaml, samples.tsv, references.yaml
src/svml/                 importable, unit-testable package
  io/                     records (pysam-free), vcf, bam, bed
  candidates/             SURVIVOR merge + provenance reconciliation
  features/               registry (schema source of truth) + 4 feature groups
  labels/                 confident-region gate + truvari matching
  ml/                     splits, dataset encoding, models, training
  eval/                   six baselines, metrics, stratification
  explain/                SHAP
  report/                 plots, HTML/PDF assembly
  cli.py                  the subcommands the Snakemake rules call
tests/                    76 tests; no bioinformatics tools required
```

Two design rules hold the architecture together:

1. **Feature extraction never reads truth data.** Truth enters only at `labels/` and
   `eval/`. That is what keeps leakage out.
2. **`features/registry.py` is the single source of truth for the schema.** The feature
   dictionary is generated from it (`svml feature-docs`), so code and docs cannot drift.

Several architectural rules are enforced in code rather than left to discipline —
`src/svml/utils/config.py` refuses to load a config that disables the confident-region
gate, marks Stage C as benchmarked, drops the union/intersection baselines, lets Stage B
see `svtype_called`, or uses different match criteria for labelling and evaluation.
`tests/test_config.py` covers each of those refusals.

---

## Prerequisites

**Windows users: WSL2 is required.** Minimap2, SAMtools, Sniffles2, cuteSV, bcftools,
SURVIVOR and Truvari have no native Windows builds. Snakemake will fail at the first
alignment rule if run from PowerShell.

```bash
wsl --install -d Ubuntu          # from PowerShell, once
```

Then, inside WSL:

```bash
mamba env create -f environment.yml
conda activate svml
```

Set `project.workdir` in `config/config.yaml` to a path inside the WSL filesystem.
`/mnt/d/...` I/O is slow enough to dominate runtime on alignment and depth queries.

---

## Running the pipeline

Inside WSL, from the repo root:

```bash
bash scripts/setup_wsl.sh               # once: micromamba + svml env + check-env
bash scripts/run_pipeline.sh milestone1 # or milestone2 / milestone3 / (none) for all
```

`setup_wsl.sh` is user-space only (no sudo) and installs everything into a single
`svml` environment, so `--use-conda` is not needed. `run_pipeline.sh` keeps a log per run
under `~/svml-work/logs/`.

Before the first real run, fill the `TODO` fields in `config/references.yaml` (truth VCF
and BED filenames from the v5.0q release listing) and `config/samples.tsv` (source URLs
from the GIAB data index). The pipeline stops with an explanatory message if they are
still `TODO`.

Milestone targets match §34, so each can be run and validated independently:

```bash
snakemake --cores 4 milestone1    # FASTQ -> QC -> BAM -> Sniffles2 + cuteSV -> candidates
snakemake --cores 4 milestone2    # candidates -> feature matrix
snakemake --cores 4 milestone3    # features -> labels -> models -> benchmark
```

Unit tests need no bioinformatics tools and run anywhere, including Windows:

```bash
python -m pytest tests/ -q
```

Everything is driven by three config files; no path is hardcoded (§21):

- `config/config.yaml` — parameters, thresholds, seeds, tool flags
- `config/samples.tsv` — sample manifest
- `config/references.yaml` — genome, annotations and truth assets with URLs and checksums

`resources/` and `results/` are git-ignored and fully reconstructible from a clean clone.
Every run writes `results/report/run_metadata.json` with tool versions, config hash,
random seeds, dataset versions and execution date (§23).

---

## Data

| Sample | Platform | Role |
|---|---|---|
| HG002 | ONT | Labelled — chromosome-grouped train/val/test |
| HG002 | PacBio HiFi | Generalisation — platform shift, same truth |
| HG005 | ONT | Unlabelled — pipeline robustness only |

Reference: GRCh38 no-alt analysis set. Truth: GIAB HG002 `v5.0q` `stvar` GRCh38.

Only HG002 has SV truth, so a sample-level split is impossible; the supervised split is
**chromosome-grouped** (train chr1–16, validate chr17–19, test chr20–22 + chrX). Reads are
acquired region-sliced from pre-aligned GIAB BAMs and converted back to FASTQ, so the full
FASTQ → Minimap2 → BAM path runs at a fraction of the whole-genome cost.

---

## Scope and limitations

In scope: human germline SVs ≥ 50 bp, ONT and PacBio HiFi, GRCh38, Minimap2, Sniffles2,
cuteSV, Random Forest, XGBoost, GIAB benchmarking, SHAP.

Known limitations, to be restated in the final report:

- No DUP / INV / TRA ground truth exists in GIAB; those classes are pseudo-labelled and
  not benchmarked.
- One labelled sample (HG002), so generalisation is tested across platforms, not samples.
- Region-scoped rather than whole-genome analysis.
- Benchmark confident regions exclude many of the hardest genomic regions, so the project
  measures a repeat-context gradient but cannot benchmark the most repetitive loci.
- `v5.0q` is a draft benchmark; putative FPs/FNs require manual curation.
- ML recall is hard-capped by the union of the two callers' candidate sets — the ML layer
  is a filter and cannot discover SVs neither caller proposed.

This is a research-grade prototype. It is **not** clinically validated and makes no medical
claims.

---

## Course alignment

| CO | Evidence |
|---|---|
| CO1 | FASTQ, SAM/BAM, VCF, BED and feature matrices |
| CO2 | Minimap2, Sniffles2, cuteSV, SAMtools, NanoPlot, Truvari |
| CO3 | Minimap2 alignment and SV signature interpretation |
| CO4 | Snakemake reproducible end-to-end pipeline |

## References

Sedlazeck et al. 2018 (Nature Methods) · Heller & Vingron 2019 (Bioinformatics) ·
Jiang et al. 2020 (Genome Biology) · Zook et al. 2020 (Nature Biotechnology) ·
Lin et al. 2022 (Nature Methods) · Popic et al. 2023 (Nature Methods) ·
Smolka et al. 2024 (Nature Biotechnology) · Li 2018 (Bioinformatics)
