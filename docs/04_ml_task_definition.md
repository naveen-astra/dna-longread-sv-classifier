# ML Task Definition

Required by PROJECT MASTER CONTEXT §13, which states the choice between targets A, B and C
must be made explicitly and justified rather than silently assumed.

**Chosen: Option C (two-stage), with stages re-weighted so every benchmarked number rests
on real GIAB labels.**

---

## The research question, restated operationally

> Can a machine-learning pipeline score candidate SV signatures and classify them into SV
> types with higher precision than heuristic calling alone, particularly in difficult /
> repetitive genomic regions?

Two measurable claims are embedded here:

1. **Precision claim** — the ML layer removes false positives that Sniffles2 and cuteSV
   emit, at an acceptable cost in recall.
2. **Region claim** — the gain is larger in repeat-rich regions than in easy regions.

Claim 1 is a **validity** task, not a type task. Claim 2 is a **stratification** of that
same task. Neither is answered by five-class type classification. This is why the task
below is weighted the way it is.

---

## Why not Option A (five-class only)

Two independent reasons, both established in `00_architecture_evaluation.md`:

**No labels exist.** Verified against the GIAB release tree: `NIST_SV_v0.6` is GRCh37 and
its README states "All now are DEL or INS"; the GRCh38-native `v5.0q` `stvar` benchmark is
assembly-derived and excludes inversion interiors from its confident regions. Neither
provides trustworthy DUP/INV/TRA labels. Building a benchmarked five-class classifier
against GIAB would require fabricating labels for 3 of 5 classes, which §28 forbids.

**It would not answer the question.** Even with perfect labels, type classification is
close to a no-op: the candidate VCF already carries `SVTYPE`, and `SVLEN`'s sign encodes
DEL vs INS. A model given those features and asked for the type learns the identity
function. A high five-class F1 would be a leakage artefact, not evidence about precision
in repeats.

## Why not Option B (validity only)

Validity alone answers the precision claim but drops the five-class deliverable that the
confirmed presentation and O3 promise. There is no need to drop it — the type stage can
still be built and reported, as long as its labels' provenance is stated.

## Why Option C, re-weighted

Option C keeps both, and the re-weighting ensures the honest task carries the headline
claim while the promised deliverable still ships with an accurate label.

---

## Task specification

### Stage A — Candidate validity. PRIMARY. Fully benchmarked.

| | |
|---|---|
| Input | One merged candidate SV + its feature vector |
| Target | `is_true ∈ {0, 1}` |
| Label source | `truvari bench` against GIAB v5.0q `stvar` GRCh38, gated on confident regions |
| Population | Merged candidates **inside** confident regions only |
| Models | Random Forest, XGBoost |
| Output | Calibrated P(true) |
| Answers | Precision claim (1) and, stratified, region claim (2) |

This is the experiment the project's conclusions rest on.

### Stage B — SV type, DEL vs INS. SECONDARY. Fully benchmarked.

| | |
|---|---|
| Input | Candidates scored true by Stage A |
| Target | `true_sv_type ∈ {DEL, INS}` |
| Label source | `SVTYPE` of the matched GIAB truth record (real label) |
| Feature restriction | Group V feature `svtype_called` **excluded**; `log10_svlen` uses absolute value (see `03_feature_dictionary.md`) |
| Models | Random Forest, XGBoost |
| Output | Calibrated class probabilities |

Two classes because those are the two the benchmark actually labels. With V2 excluded this
is a genuine prediction task rather than an echo of the caller.

### Stage C — Five-class INS/DEL/DUP/INV/TRA. EXPLORATORY. Not benchmarked.

| | |
|---|---|
| Input | All merged candidates |
| Target | `pseudo_sv_type ∈ {INS, DEL, DUP, INV, TRA}` |
| Label source | **Caller-consensus pseudo-labels** — candidates where Sniffles2 and cuteSV agree on type. **Not ground truth.** |
| Purpose | Delivers the five-class artefact O3/§10 promise; characterises DUP/INV/BND candidate behaviour |
| Reporting rule | Every Stage C number carries the label "pseudo-label derived; not GIAB-benchmarked" in the same table cell or figure caption, never only in a footnote |

Stage C measures **agreement with the callers' consensus**, which is a different and much
weaker quantity than accuracy. It is reported for completeness and is explicitly excluded
from any claim of improvement over the callers. Per-class results for TRA in particular
should be read as descriptive only — HG002 is a healthy genome with no known balanced
translocation, so the TRA candidate set is expected to be near-entirely artefactual.

---

## Restated objective O3

Original: *five-class classifier, macro-F1 ≥ 0.85 on held-out data.*

Restated:

- **O3a** — Stage A validity classifier, **macro-F1 ≥ 0.85** on held-out chromosomes.
- **O3b** — Stage B DEL/INS classifier, **macro-F1 ≥ 0.85** on held-out chromosomes.
- **O3c** — Stage C five-class model delivered and reported, with pseudo-label provenance
  stated. **No F1 target**, because the labels are not ground truth.

Both targets remain aspirational, not guaranteed (§16, §28). If they are not met, the
measured value is reported as-is with error analysis.

---

## Labelling rules

The confident-region gate (finding H4) is what makes the negative class trustworthy.

```
for each merged candidate:
    if candidate NOT inside confident-region BED:
        is_true = NA          # benchmark makes no claim here
        excluded from train AND test; carried to inference; counted in report
    elif truvari matches a truth record:
        is_true = 1
        true_sv_type = truth record SVTYPE
    else:
        is_true = 0           # inside a confident region with no truth record => FP
        true_sv_type = NA
```

The critical rule: **absence of a truth match outside confident regions is not evidence of
falsehood.** Labelling those candidates 0 would teach the model that real SVs in hard
regions are artefacts — precisely inverting the project's goal. Counts in all three buckets
are reported.

`truvari bench` parameters (`--pctseq`, `--pctsize`, `--refdist`) live in `config.yaml`. The
**same** parameters are used for labelling and for benchmarking, so the labeller and the
evaluator cannot disagree about what a match is.

---

## Split strategy and leakage control

Only HG002 has SV truth, so a sample-level split is impossible (finding H3). The split is
**chromosome-grouped**:

| Split | Chromosomes | Use |
|---|---|---|
| Train | chr1–chr16 | Model fitting, imputation fitting |
| Validation | chr17, chr18, chr19 | Hyperparameter search, threshold selection, calibration |
| Test | chr20, chr21, chr22, chrX | Touched **once**, at the very end |
| Pilot | chr21 or chr22 | Pipeline development — see note below |

**Pilot design, revised after contact with the data (see `06_execution_plan.md`, D5).**
The original plan piloted on chr19 alone. That cannot work: the split is grouped by
chromosome, so a one-chromosome pilot leaves two splits empty and nothing can be trained.
The pilot therefore uses windows on one chromosome per split:

| Split | Window(s) |
|---|---|
| train | chr16:1–15 Mb, chr16:85–90 Mb |
| val | chr19:1–10 Mb |
| test | chr22:40–50 Mb |

The test window is still touched exactly once, at the final benchmark, after models and
thresholds are frozen on validation. Pipeline *debugging* happens on the train and
validation windows. The windows are SV-rich sub-telomeric regions chosen to fit the
download budget, so pilot metrics describe those regions, not the genome.

Leakage controls beyond the split:

1. **Merge before split.** Sniffles2 and cuteSV calls at the same locus are merged into one
   candidate before splitting, so the same event cannot appear in two splits.
2. **Chromosome grouping** means no genomic locus spans splits (§11).
3. **Raw coordinates excluded** as features — `chrom` is the grouping variable and `POS` is
   an arbitrary index. Only derived context (Group A) is used.
4. **Preprocessing fit on train only**, inside a `Pipeline`.
5. **Test set touched once.** All tuning happens on validation. `tests/test_splits.py`
   asserts the chromosome-disjointness invariant in CI.

## Class imbalance

Expected to be substantial — merged long-read candidate sets typically contain far more
true DEL/INS than false ones inside confident regions, and the ratio shifts with repeat
context. Handling, in order of preference:

1. Report the observed class distribution before choosing any remedy.
2. `class_weight="balanced"` (RF) / `scale_pos_weight` (XGBoost).
3. Threshold tuning on the validation split, optimising the operating point rather than
   resampling.
4. Resampling (SMOTE etc.) **only if** 1–3 are insufficient, and never applied to the
   validation or test splits.

Because the benchmark comparison is done with PR curves (finding H2), threshold choice does
not distort the headline comparison.

## Generalisation test

HG002 PacBio HiFi as a held-out **platform shift** set: same truth, different error
profile. Models trained on ONT are evaluated on HiFi candidates without retraining. This is
a real generalisation test and does not require a second truth set. The third sample
(HG005 ONT) has no SV truth and is used only to demonstrate pipeline robustness; it is
reported as unlabelled.
