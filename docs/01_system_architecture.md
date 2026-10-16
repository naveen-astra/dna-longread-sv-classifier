# System Architecture (revised)

Supersedes the diagram in PROJECT MASTER CONTEXT §4. Changes are driven by the findings in
`00_architecture_evaluation.md`; each new or moved box is tagged with the finding that
caused it.

---

## A. System architecture

```
STAGE 0  REFERENCE & ANNOTATION PREPARATION                            [M1]
┌──────────────────────────────────────────────────────────────────────────┐
│ GRCh38 no-alt analysis set ──► samtools faidx ──► minimap2 -d index.mmi  │
│ UCSC rmsk / simpleRepeat / genomicSuperDups / gaps ──► normalised BEDs   │
│ GIAB v5.0q stvar GRCh38  ──► truth VCF + confident-region BED      [B2]  │
│ All assets pinned in config/references.yaml (url, version, md5, build)   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
─── RAW DATA ───────────────────────┼──────────────────────────────────────
                                    ▼
STAGE 1  DATA ACQUISITION
┌──────────────────────────────────────────────────────────────────────────┐
│ GIAB pre-aligned BAM ──► samtools view -b <region> ──► region slice      │
│                      ──► samtools fastq ──► region-sliced FASTQ    [M2]  │
│ 3 datasets: HG002-ONT, HG002-HiFi, HG005-ONT   (manifest: samples.tsv)   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
─── BIOINFORMATICS PROCESSING ──────┼──────────────────────────────────────
                                    ▼
STAGE 2  QUALITY CONTROL                    NanoPlot ──► qc_metrics.json
                                    │        read count, N50, length & Q dist
                                    ▼
STAGE 3  ALIGNMENT                          minimap2 -ax map-ont | map-hifi
                                    │       samtools sort / index / stats
                                    ▼
                          ┌──────────────────────┐
                          │ Sorted + indexed BAM │
                          └──────────┬───────────┘
                                     │
─── SV CANDIDATE GENERATION ─────────┼─────────────────────────────────────
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
          ┌─────────────────┐               ┌─────────────────┐
          │ Sniffles2       │               │ cuteSV          │
          └────────┬────────┘               └────────┬────────┘
                   │                                 │
                   ▼                                 ▼
          ┌──────────────────────────────────────────────────┐
          │ bcftools norm  +  filter SVLEN >= 50bp     [H5]  │
          └────────────────────────┬─────────────────────────┘
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ truvari collapse merge  (params in config)[H5]   │
          │ ──► unified candidate set, per-caller provenance │
          └────────────────────────┬─────────────────────────┘
                                   │
─── FEATURE ENGINEERING ───────────┼───────────────────────────────────────
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ FEATURE EXTRACTION                               │
          │   Group B  BAM evidence   (pysam)                │
          │   Group A  annotation context (BED overlap)      │
          │   Group C  caller-reported fields                │
          │   Group V  caller-declared type / length         │
          └────────────────────────┬─────────────────────────┘
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ FEATURE MATRIX  (parquet)  1 row = 1 candidate   │
          │ >= 15 features, schema in 03_feature_dictionary  │
          └────────────────────────┬─────────────────────────┘
                                   │
─── LABELLING ─────────────────────┼───────────────────────────────────────
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ CONFIDENT-REGION GATE                      [H4]  │
          │   inside  ──► truvari bench vs v5.0q stvar       │
          │             match    ──► is_true = 1             │
          │             no match ──► is_true = 0             │
          │   outside ──► is_true = NA  (excluded from       │
          │               train AND test; reported separately)│
          └────────────────────────┬─────────────────────────┘
                                   │
─── MACHINE LEARNING ──────────────┼───────────────────────────────────────
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ CHROMOSOME-GROUPED SPLIT                   [H3]  │
          │ train chr1-16 | val chr17-19 | test chr20-22,X   │
          └────────────────────────┬─────────────────────────┘
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐
          │ Random Forest   │           │ XGBoost         │
          └────────┬────────┘           └────────┬────────┘
                   └──────────────┬──────────────┘
                                  ▼
          ┌──────────────────────────────────────────────────┐
          │ Stage A  validity score  (PRIMARY)         [B1]  │
          │ Stage B  DEL vs INS type (SECONDARY)             │
          │ Stage C  5-class, pseudo-labels (EXPLORATORY)    │
          │ + ablation: B+A  vs  B+A+C+V               [M3]  │
          └────────────────────────┬─────────────────────────┘
                                   │
─── BENCHMARKING ──────────────────┼───────────────────────────────────────
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ truvari bench --refine  vs  GIAB v5.0q     [H5]  │
          │ Baselines: Sniffles2 | cuteSV | union |          │
          │            intersection | RF | XGBoost     [H2]  │
          │ PR curves + matched-recall / matched-precision   │
          │ Stratified by repeat context               [L1]  │
          └────────────────────────┬─────────────────────────┘
                                   │
─── EXPLAINABILITY ────────────────┼───────────────────────────────────────
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ SHAP: global importance, beeswarm, per-prediction│
          └────────────────────────┬─────────────────────────┘
                                   │
─── REPORTING ─────────────────────┼───────────────────────────────────────
                                   ▼
          ┌──────────────────────────────────────────────────┐
          │ Final annotated VCF + HTML/PDF report + JSON     │
          │ + run_metadata.json (tool versions, seeds, dates)│
          └──────────────────────────────────────────────────┘
```

---

## B. Data flow (artefacts, not stages)

```
references.yaml ──► GRCh38.fa ─────┬──► GRCh38.fa.fai
                                   ├──► GRCh38.mmi ──────────────┐
                                   └──► (used by callers) ───────┤
                   annotations/*.bed ──────────────────┐         │
                   truth/v5.0q.stvar.GRCh38.vcf.gz ──┐ │         │
                   truth/v5.0q.stvar.GRCh38.bed ─────┤ │         │
                                                     │ │         │
samples.tsv ──► raw/{sample}.region.fastq.gz ────────┼─┼─────────┤
                     │                               │ │         │
                     ├──► qc/{sample}/NanoPlot/ ──► qc_metrics.json
                     │                               │ │         │
                     └──► align/{sample}.sorted.bam ─┼─┼─────────┘
                              │      .bai            │ │
                              │      align_stats.txt │ │
                              │                      │ │
              ┌───────────────┴───────────────┐      │ │
              ▼                               ▼      │ │
   sv/{sample}.sniffles.vcf.gz   sv/{sample}.cutesv.vcf.gz
              │                               │      │ │
              └───────────────┬───────────────┘      │ │
                              ▼                      │ │
              sv/{sample}.norm.*.vcf.gz              │ │
                              ▼                      │ │
              candidates/{sample}.merged.vcf.gz      │ │
                              │                      │ │
                              ├──── + BAM + annotations ──► features/{sample}.parquet
                              │                      │ │
                              └──── + truth VCF + confident BED ──► labels/{sample}.parquet
                                                       │
            features ⋈ labels ──► ml/dataset.parquet ──┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
          models/rf_{stage}.pkl  models/xgb_{stage}.pkl  ml/splits.json
                     │                  │
                     └────────┬─────────┘
                              ▼
                  predictions/{sample}.parquet
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
 results/final.vcf.gz  results/metrics.json  results/shap/*.png
        └─────────────────────┼─────────────────────┘
                              ▼
                     report/report.html
                     report/summary.json
                     report/run_metadata.json
```

---

## C. ML data flow

```
                    unified candidate set
                            │
                            ▼
              ┌─────────────────────────────┐
              │ feature extraction          │
              │  B  BAM-derived             │
              │  A  annotation-derived      │
              │  C  caller-reported         │
              │  V  caller-declared type    │
              └──────────────┬──────────────┘
                             ▼
                      feature matrix
                             │
                             ▼
              ┌─────────────────────────────┐
              │ confident-region gate  [H4] │
              └──┬───────────┬──────────┬───┘
      is_true=1  │  is_true=0│   is_true=NA
                 ▼           ▼          ▼
              ┌──────────────────┐   ┌──────────────────────┐
              │ labelled set     │   │ unlabelled stratum   │
              │ (train/val/test) │   │ inference only,      │
              └────────┬─────────┘   │ counted, not scored  │
                       │             └──────────────────────┘
                       ▼
          GroupKFold by chromosome  [H3]  — no locus spans splits
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
       train          val           test
         │             │             │
         │  hyperparam search        │  touched ONCE, at the end
         │  (val only)               │
         ▼                           │
   ┌──────────────────────────┐      │
   │ STAGE A  is_true (binary)│      │
   │ RF and XGBoost           │      │
   └────────┬─────────────────┘      │
            │ candidates scored true │
            ▼                        │
   ┌──────────────────────────┐      │
   │ STAGE B  DEL vs INS      │      │
   │ RF and XGBoost           │      │
   └────────┬─────────────────┘      │
            │                        │
            ▼                        ▼
   ┌──────────────────────────────────────────┐
   │ STAGE C  5-class, caller-consensus       │
   │ PSEUDO-LABELS — exploratory, never       │
   │ reported as benchmarked ground truth [B1]│
   └──────────────────┬───────────────────────┘
                      ▼
            calibrated probabilities
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   truvari bench   PR curves     SHAP
   vs baselines   + strata    (global + local)
```

---

## Layer boundaries

| Layer | Owns | Must not |
|---|---|---|
| Reference prep | genome, index, annotations, truth | see sample reads |
| Acquisition | manifest, region slicing | modify raw data (§9) |
| QC | metrics, accept/flag decision | silently drop reads |
| Alignment | BAM + stats | assume a hardcoded reference |
| SV calling | per-caller VCFs | be fed into the model directly (§5) |
| Merging | unified candidates + provenance | concatenate blindly |
| Features | feature matrix | read truth data (leakage) |
| Labelling | labels + confident-region gate | assign a label outside confident regions |
| ML | models, predictions | see the test chromosomes before final eval |
| Benchmark | metrics vs all 6 baselines | use a bespoke matcher |
| Explain | SHAP | alter predictions |
| Report | HTML/PDF/JSON | present illustrative values as results (§28) |

The critical invariant: **the feature-extraction layer never touches truth data.** Truth
enters only at the labelling and benchmarking layers. This is what keeps leakage out.
