# Architecture Evaluation

Status: Phase 1 (architecture). No pipeline code written yet, by design (spec §25, §26).
Date of evaluation: 2026-09-22.

This document evaluates the architecture in the PROJECT MASTER CONTEXT against what the
named data sources and tools can actually deliver. Findings are ordered by severity.
Every blocking finding below was verified against the live GIAB FTP release tree, not
assumed.

---

## Verdict

The **bioinformatics half** of the architecture (§4 top: FASTQ → QC → Minimap2 → BAM →
Sniffles2/cuteSV → candidates) is sound and standard. Keep it essentially as drawn.

The **machine-learning half** has one blocking defect and several design weaknesses. The
blocking defect is that **objective O3 — a five-class INS/DEL/DUP/INV/TRA classifier
benchmarked against GIAB at macro-F1 ≥ 0.85 — cannot be built as specified, because no
GIAB truth set labels DUP, INV or TRA.** This is not a tuning problem or a compute
problem; the labels do not exist.

Everything else is fixable without changing the project's stated scope, contribution, or
course-outcome alignment. §13 of the spec already anticipates the need to pin down the ML
target and authorises the agent to choose and document one, so the fix below is within
the spec rather than a departure from it.

---

## B1 — BLOCKING: the five-class target has no ground truth

Verified on the GIAB FTP tree:

| Benchmark | Reference build | SV content |
|---|---|---|
| `NIST_SV_v0.6/HG002_SVs_Tier1_v0.6.vcf.gz` | **GRCh37 only** | "No variants have SVTYPE=COMPLEX. **All now are DEL or INS.**" (README_SV_v0.6.txt). ~9,600 PASS Tier1 SVs in confident regions. README states "GRCh38 callsets are under development". |
| `v5.0q/` (Q100 / assembly-based, `stvar`) | GRCh37, **GRCh38**, CHM13v2.0 | Sequence-resolved SVs from a diploid assembly. Does not exclude complex SVs, but the benchmark regions **exclude the interiors of inversions, "even if they are very large"**. |
| `NISTv4.2.1/` | GRCh38 | **Small variants (SNV/indel), not SVs.** |

Three consequences.

**(a) DUP, INV and TRA have no labels.** v0.6 is DEL/INS by construction. v5.0q is
assembly-derived and represents variation as sequence-resolved allele replacements, and
explicitly carves inversion interiors out of the confident regions. Neither provides a
trustworthy `true_sv_type ∈ {DUP, INV, TRA}`. Training a five-class model against GIAB
would require inventing labels for three of the five classes — which §28 of the spec
forbids ("call an ML target 'ground truth' if it came from the same heuristic caller").

**(b) Germline TRA is near-absent in HG002 anyway.** HG002 is a healthy reference
individual with no known balanced translocation. Sniffles2 emits `BND` records rather
than `SVTYPE=TRA`, and in a normal germline genome the overwhelming majority of BND calls
are alignment artefacts. A "TRA" class would be a class of almost pure false positives.
Even with labels, per-class F1 for TRA would be meaningless.

**(c) The spec cites the wrong URL.** §7 points the ground truth at
`.../NISTv4.2.1/GRCh38/`, which is the **small-variant** benchmark. The SV benchmark is a
different directory. This needs correcting in the manifest or the whole labelling stage
targets the wrong file.

### Fix

Adopt §13 **Option C (two-stage)**, with the stages re-weighted so the benchmarked claims
only ever rest on real labels:

- **Stage A — candidate validity (`is_true`), binary. PRIMARY, fully benchmarked.**
  Does this candidate correspond to a real SV? This is the task that actually answers the
  research question ("higher precision than heuristic calling"), and it is fully labelled
  by GIAB.
- **Stage B — SV type, DEL vs INS. SECONDARY, fully benchmarked.** Two classes, real
  labels, honest metrics.
- **Stage C — five-class INS/DEL/DUP/INV/TRA. EXPLORATORY, explicitly not benchmarked
  as ground truth.** Trained on caller-consensus pseudo-labels, reported with the
  pseudo-label provenance stated in the same table as the numbers.

This keeps the five-class deliverable the presentation promised, while ensuring no
benchmarked precision/recall/F1 number is ever computed against a fabricated label.

**Restate O3 as:** macro-F1 ≥ 0.85 on Stage A (validity) and Stage B (DEL/INS type),
held out by chromosome. Stage C reported descriptively with pseudo-label caveats.
See `04_ml_task_definition.md`.

---

## B2 — BLOCKING: truth build conflicts with the mandated reference

The spec mandates GRCh38 (§2, §6). The classic SV benchmark (v0.6) is **GRCh37 only**.
Using it would force either an unofficial liftover of the truth VCF and its confident-region
BED — liftover of SV coordinates and interval sets is lossy and itself a source of false
FP/FN — or a switch of the whole project to GRCh37.

### Fix

Use **`v5.0q` `stvar` on GRCh38** as the primary truth set. It is GRCh38-native, so no
liftover is needed, and the mandated reference build is preserved. Record v0.6/GRCh37 as
an optional secondary sanity check only, clearly labelled as a different build.

Note the tradeoff and document it: v5.0q is a draft benchmark and does not exclude complex
SVs, so its README recommends comparing with `truvari --refine`, `vcfdist` or `hap-eval`
and manually curating a subset of putative FPs/FNs. That curation step is now a required
part of the evaluation protocol rather than an optional extra.

---

## H1 — HIGH: type classification is close to a no-op, and is the wrong headline task

If the model is fed the candidate's own `SVTYPE` (or `SVLEN`, whose sign encodes DEL vs
INS) and asked to predict the type, it learns the identity function. A 5-class accuracy
near 1.0 would be a leakage artefact, not a result.

Type is not the hard part of long-read SV calling. Sniffles2 and cuteSV already assign
type from alignment signatures with high reliability for DEL/INS. **Validity is the hard
part** — the callers' weakness is false positives in repeats, which is exactly what the
research question names.

### Fix

Make Stage A (validity) the headline experiment, as in B1. For Stage B, either exclude
the caller's own `SVTYPE` from the feature set, or keep it and report the ablation
(with/without caller-declared type) so the contribution of the BAM-derived evidence is
visible. Both are specified in `03_feature_dictionary.md` as feature groups `C` and `V`.

---

## H2 — HIGH: the benchmark comparison as drawn is not a fair fight

The architecture compares "RF / XGBoost" against "Sniffles2 / cuteSV" on precision, recall
and F1. But in this design the ML layer is a **filter over the union of the two callers'
candidates**. It can only ever remove candidates. Therefore:

- ML recall is **hard-capped** at the recall of the union candidate set.
- ML will essentially always look better on precision and worse on recall than the union.
- Reporting a single operating point can make the ML layer look arbitrarily good or bad
  depending on the threshold chosen.

Comparing a filter to a caller at one threshold is not a meaningful comparison, and §28
forbids claiming improvement that the benchmark does not demonstrate.

### Fix

Fix the comparison protocol (details in `05_evaluation_protocol.md`):

1. Add the two missing baselines that are the ML layer's real competitors:
   **union** (recall ceiling) and **intersection / caller-agreement** (the trivial
   heuristic filter that ML must beat).
2. Report the full **precision–recall curve** for RF and XGBoost, not one point.
3. Compare at **matched recall** and at **matched precision** against each baseline, and
   state the union recall ceiling explicitly next to every ML recall number.
4. Report ΔF1 against the intersection baseline as the headline contribution figure.

Baseline set becomes: Sniffles2 alone, cuteSV alone, union, intersection, RF, XGBoost.

---

## H3 — HIGH: sample-level splitting is impossible; only HG002 has SV truth

§11 offers "sample-level split" as the preferred anti-leakage strategy. But the GIAB SV
truth used here exists for **HG002 only**. There is no second labelled sample to hold out,
so a sample-level split cannot be constructed for the supervised task.

Separately, O1 asks for QC-checked BAMs for **at least 3 public samples**, which reads as
if three labelled samples were expected.

### Fix

Separate the two requirements — O1 is a bioinformatics objective, not an ML one, and it
says "3 public samples", not "3 labelled samples":

- **O1 (3 samples, BAM-level):** HG002 ONT, HG002 PacBio HiFi, and one additional GIAB
  sample (HG005 or HG001) ONT. All three go through QC → Minimap2 → BAM → both callers.
  Satisfies O1 literally and demonstrates CO1–CO3.
- **Supervised ML:** HG002 only, split **by chromosome** (`GroupKFold` on chromosome),
  because near-identical candidates from the same locus must not straddle the split.
- **Generalisation test:** HG002 PacBio HiFi as a *platform-shift* held-out set (same
  truth, different error profile). This is a genuinely informative generalisation test
  and does not require a second truth set.
- The third sample is used for candidate generation and pipeline-robustness demonstration
  only, and is reported as unlabelled. Say so in the report.

Proposed chromosome split: train chr1–chr16, validate chr17–chr19, test chr20–chr22 + chrX.
chr20 is also the pilot region, so reserve it for test and pilot on chr21 or chr22 to avoid
tuning on the test chromosome.

---

## H4 — HIGH: labelling needs a confident-region gate, and it is not in the architecture

The architecture goes "candidates → feature extraction → ML" with truth entering only at
the benchmarking box near the bottom. But labels must be created *before* training, and
naive labelling silently poisons the negative class.

A candidate that does not match the truth VCF is **not automatically a false positive**.
It is a false positive only if it lies **inside the benchmark's confident regions**. A
candidate outside those regions is simply *unknown* — the benchmark makes no claim there.
Labelling unknowns as negatives teaches the model that real SVs in hard regions are
artefacts, which would actively defeat the project's stated goal of performing well in
difficult/repetitive regions.

### Fix

Add an explicit **confident-region gate** as its own stage between candidate merging and
label assignment, with a three-way outcome:

- inside confident regions **and** matches truth → `is_true = 1`
- inside confident regions **and** no match → `is_true = 0`
- outside confident regions → `is_true = NA`, **excluded from train and from test**,
  carried through to inference and reported as an unlabelled/unscored stratum

Report the count in each bucket. It is a legitimate and important limitation figure.

---

## H5 — HIGH: merging and matching must use standard tools, not bespoke code

§5 and §12 ask for a hand-rolled merge rule and a hand-rolled truth matcher (coordinate
tolerance, size similarity, type compatibility). SV comparison is notoriously subtle —
the same deletion can be represented at different positions with different sequence in
two VCFs and still be the same event. Hand-rolled matchers are the classic source of
inflated or deflated benchmark numbers, and a reviewer will not trust one.

### Fix

Delegate both to the community-standard tools, and keep the tuned parameters in config:

- **Normalisation:** `bcftools norm` + a filter to SVs ≥ 50 bp.
- **Caller merge:** `SURVIVOR merge` or `Jasmine` (or `truvari collapse`), with
  distance/type/size parameters recorded in `config.yaml`.
- **Truth matching (labels) and benchmarking (metrics):** **`truvari bench`**, and for
  v5.0q specifically, `truvari bench --refine` as the benchmark README recommends.

Using the same tool for label creation and for evaluation also removes a subtle
inconsistency: if the labeller and the evaluator disagree about what counts as a match,
the reported metrics measure the mismatch rather than the model.

---

## M1 — MEDIUM: reference and annotation data are missing from the architecture

The diagram starts at FASTQ. Several required inputs are never acquired:

- **GRCh38 no-alt analysis set.** Use `GCA_000001405.15_GRCh38_no_alt_analysis_set.fna`.
  Aligning long reads to a reference *with* ALT contigs fragments alignments and produces
  spurious SV signatures. This choice must be explicit, not incidental.
- **Minimap2 index** (`.mmi`) build step, plus `samtools faidx`.
- **Repeat / context annotations**, required for the mandated `repeat_context` feature:
  UCSC `rmsk`, `simpleRepeat`, `genomicSuperDups`, and an assembly-gap/N-run track.
- **Confident-region BED** from the chosen truth release.

### Fix

Add a **Stage 0 — Reference & Annotation Preparation** to the architecture, with its own
manifest (`config/references.yaml`) carrying URL, version, checksum and build for every
asset. This also satisfies the §23 reproducibility requirement, which currently has no
home in the pipeline.

---

## M2 — MEDIUM: whole-genome scope collides with the stated compute budget

§24 assumes limited storage and campus compute; O1 asks for 3 samples aligned to GRCh38.
Whole-genome ONT at 30–50× is roughly 100–200 GB of FASTQ and 50–100 GB of BAM **per
sample**, plus many CPU-hours of Minimap2. Three samples at WGS is not compatible with
§24, and re-basecalling or re-aligning full genomes adds nothing scientifically here.

### Fix

Two-tier data strategy, encoded in config rather than improvised:

- **Tier 1 (pilot):** one chromosome (chr21 or chr22 — *not* chr20, which is reserved for
  test). Region-sliced FASTQ. Validates every stage end to end.
- **Tier 2 (full run):** the chromosome set required by the split in H3.

Acquire reads as **region-sliced** data: download GIAB's pre-aligned BAM, slice the target
region with `samtools view -b <region>`, and convert back to FASTQ for the Minimap2 stage.
This preserves the genuine FASTQ → Minimap2 → BAM path that CO2/CO3 require as evidence,
at a small fraction of the storage and runtime. Document that alignment is region-scoped.

---

## M3 — MEDIUM: caller-internal fields make "we beat the callers" a weak claim

Features like Sniffles2 `SUPPORT`, cuteSV `RE`, and caller-agreement count are strong
predictors, but they are the callers' own internal evidence. A model that wins mainly on
those is closer to a recalibration of the callers than to independent evidence.

### Fix

Partition features into groups and run a **required ablation**:

- **Group B** — BAM-derived, computed independently of the callers
- **Group A** — annotation-derived
- **Group C** — caller-reported fields
- **Group V** — caller-declared type/length

Train and report at minimum `B+A` and `B+A+C+V`. The `B+A` model is the honest measure of
whether independent evidence adds anything. Both go in the report.

---

## M4 — MEDIUM: Windows host cannot run the toolchain natively

Minimap2, SAMtools, Sniffles2, cuteSV, bcftools, SURVIVOR and Truvari do not run natively
on Windows. The development machine is Windows 11. Snakemake `--cores 4` will fail at the
first alignment rule.

### Fix

Make the execution environment explicit in the architecture: **WSL2 (Ubuntu) + Mamba**, or
Docker. Keep the project on the Windows filesystem but run the pipeline from WSL; note
that `/mnt/d/...` paths are slow for I/O-heavy steps, so the working/scratch directory
should live inside the WSL filesystem while results are written back. Record this in the
README as a prerequisite, not a footnote.

---

## L1 — LOW: research question vs benchmark coverage

The research question targets "difficult/repetitive genomic regions", but benchmark
confident regions systematically *exclude* many of the hardest ones (v0.6 excluded large
tandem repeats and segdups; v5.0q excludes inversion interiors).

So the project can honestly measure performance in *moderately* difficult regions inside
the confident set, but cannot benchmark the hardest regions — there is no truth there.

### Fix

Do not weaken the research question, but make it measurable. Stratify every reported
metric by repeat context (`tr_overlap_frac`, `segdup_overlap`) so the precision gain in
the repeat-rich stratum *inside* the confident regions is quantified. That is a real,
defensible answer to the research question. State the exclusion of the hardest regions in
Limitations (§19 item 19).

---

## L2 — LOW: gaps in the §4 diagram

The diagram omits: Stage 0 reference/annotation prep (M1); VCF normalisation; the
confident-region gate (H4); label generation as a stage distinct from benchmarking; the
ablation branch (M3); model/run metadata capture (§23); and the inference path for
unlabelled candidates. Corrected diagrams are in `01_system_architecture.md`.

---

## Summary of changes to the spec

| # | Spec item | Change | Type |
|---|---|---|---|
| B1 | O3, §10, §13 | Two-stage target; 5-class becomes exploratory on pseudo-labels | **Blocking** |
| B2 | §6, §7 | Truth = `v5.0q` `stvar` GRCh38, not `NISTv4.2.1`; v0.6 is GRCh37/DEL+INS only | **Blocking** |
| H1 | §10 | Validity, not type, is the headline task | High |
| H2 | §15, §16 | Add union + intersection baselines; PR curves; matched-recall comparison | High |
| H3 | O1, §11 | 3 samples at BAM level; chromosome split for ML; HiFi as generalisation set | High |
| H4 | §12 | Add confident-region gate; three-way label incl. `NA` | High |
| H5 | §5, §12 | Use SURVIVOR/Jasmine + Truvari, not bespoke matchers | High |
| M1 | §4, §9 | Add Stage 0 reference & annotation prep | Medium |
| M2 | §24 | Region-sliced two-tier data strategy | Medium |
| M3 | §15 | Required feature-group ablation | Medium |
| M4 | §22 | WSL2/Docker execution layer | Medium |
| L1 | §1, §19 | Stratify metrics by repeat context; state coverage limits | Low |
| L2 | §4 | Diagram gaps | Low |

Unchanged: scope (§2), the no-new-caller principle (§5), the software stack (§22, plus
Truvari/SURVIVOR), reproducibility requirements (§23), staged development (§25), course
alignment (§31), and the stated research contribution (§29).

---

## Decisions that need your sign-off

These three change what gets presented, so they are flagged rather than assumed:

1. **O3 restated** — macro-F1 ≥ 0.85 now applies to validity + DEL/INS type. The
   five-class model still ships, but its numbers are labelled as pseudo-label based.
   If the course requires a benchmarked five-class result, that requirement cannot be met
   with GIAB and we should raise it with the instructor early rather than at submission.
2. **Truth set switched** to GIAB `v5.0q` `stvar` GRCh38, so GRCh38 is preserved.
3. **Scope reduced to selected chromosomes**, not whole genome, per §24.

---

## Sources

- [GIAB HG002 release tree](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/)
- [NIST_SV_v0.6 README](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NIST_SV_v0.6/README_SV_v0.6.txt)
- [HG002 v5.0q benchmark README](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/v5.0q/NIST_HG002_v5.0q_variant-benchmarksets_README.md)
- [Truvari discussion #117 — understanding the HG002 truthset](https://github.com/ACEnglish/truvari/discussions/117)
- Zook et al., 2020, *A robust benchmark for detection of germline large insertions and deletions*, Nature Biotechnology.
