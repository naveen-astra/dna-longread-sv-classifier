# Feature Dictionary (draft schema)

Fulfils PROJECT MASTER CONTEXT §27 and objective O2 (≥15 features).

**Status: DRAFT.** Per §27 and §36, no feature is confirmed until it has been computed on
real HG002 data and manually inspected (Phase 6). Features that turn out to be constant,
degenerate, or uncomputable at scale will be dropped, and this table updated with the
reason. `src/svml/features/registry.py` is the machine-readable source of truth; this
document is generated from it.

Count as drafted: **33 features** across 4 groups (B=15, A=8, C=7, V=3). Objective O2
requires ≥15, so there is headroom to drop under-performing ones without breaching the
objective. The `B+A` ablation alone contributes 23 features, so it satisfies O2 on its
own — which is what makes that ablation a standalone result rather than a diagnostic.

These counts are asserted in `tests/test_features.py`, and the table below is generated
from `src/svml/features/registry.py` via `svml feature-docs`.

## Feature groups and the ablation (finding M3)

| Group | Source | Independent of the callers? | In `B+A` model | In `B+A+C+V` model |
|---|---|---|---|---|
| **B** | BAM alignment evidence | Yes | ✓ | ✓ |
| **A** | Reference annotations | Yes | ✓ | ✓ |
| **C** | Caller-reported evidence fields | No | ✗ | ✓ |
| **V** | Caller-declared type / length | No | ✗ | ✓ |

The `B+A` model is the honest test of whether independently-derived evidence adds anything
beyond what the callers already report. Both models are trained and both are reported.

Window parameters (`W_flank`, `W_bp`) live in `config.yaml`; defaults below are starting
points to be tuned on the validation chromosomes only.

---

## Group B — BAM-derived (pysam, computed from alignments alone)

| # | Feature | Definition | Calculation | Type | Range | Missing | Biological meaning / why useful |
|---|---|---|---|---|---|---|---|
| B1 | `dp_local` | Local read depth at the breakpoint | Mean per-base depth in ±`W_bp` (default 200 bp) around POS | float | ≥0 | 0 if no reads | Low depth → weak evidence for any call; very high depth flags collapsed repeats |
| B2 | `dp_flank_ratio` | **Required `read_depth_ratio`.** Depth inside the SV span ÷ mean depth of both flanks | mean depth(POS..END) / mean(depth(left flank), depth(right flank)), flanks = `W_flank` (default 1 kb) | float | ≥0 | NaN for INS/BND (no span) + indicator | Direct copy-number signal: ~0.5 or 0 for true DEL, >1.3 for true DUP. The single most discriminative depth feature |
| B3 | `dp_mad_local` | Depth instability near the breakpoint | Median absolute deviation of per-base depth in the ±`W_bp` window | float | ≥0 | 0 | Erratic depth indicates mapping instability, common at artefactual calls |
| B4 | `mean_mapq` | **Required MAPQ feature.** Mean mapping quality of reads overlapping the breakpoint | Mean of `read.mapping_quality` over overlapping reads | float | 0–60 | NaN if no reads | Low MAPQ = ambiguous placement; artefacts concentrate here |
| B5 | `median_mapq` | Median mapping quality | As B4, median | float | 0–60 | NaN if no reads | Robust to a few high-MAPQ outliers that skew the mean |
| B6 | `frac_mapq0` | Fraction of overlapping reads with MAPQ = 0 | count(MAPQ==0) / count(reads) | float | 0–1 | NaN if no reads | Direct, annotation-free measure of repetitiveness; complements Group A |
| B7 | `n_split_reads` | Reads with a supplementary alignment spanning the breakpoint | Count of reads carrying an `SA` tag whose segments flank POS | int | ≥0 | 0 | Split reads are the primary long-read signature of DEL/DUP/INV/BND |
| B8 | `n_clipped_reads` | Soft- or hard-clipped reads at the breakpoint | Count of reads with a clip ≥ `min_clip` (default 50 bp) within ±`W_bp` of POS | int | ≥0 | 0 | Clipping clusters at real breakpoints; scattered clipping suggests noise |
| B9 | `clip_frac` | Clipped fraction | `n_clipped_reads` / `dp_local` | float | 0–1 | NaN if depth 0 | Normalises B8 for coverage, making it comparable across loci and samples |
| B10 | `n_spanning_reads` | Reads crossing the whole span with no large indel in CIGAR | Reads where aligned block covers POS..END with no I/D ≥ 50 bp | int | ≥0 | 0 | Reference-supporting reads. Many spanning reads with few supporting reads → likely FP |
| B11 | `support_ratio` | Allele-fraction proxy | supporting reads / (supporting + `n_spanning_reads`) | float | 0–1 | NaN if denominator 0 | Germline SVs cluster near 0.5 (het) or 1.0 (hom). Values near 0.1 are typically artefacts |
| B12 | `strand_balance` | Strand asymmetry of supporting reads | \|n_fwd − n_rev\| / (n_fwd + n_rev) | float | 0–1 | NaN if no support | Real germline SVs show balanced strands; strong bias indicates a systematic artefact |
| B13 | `mean_read_len_local` | Mean aligned read length at the locus | Mean `reference_length` of overlapping reads | float | >0 | NaN if no reads | Short local reads reduce the chance of spanning a large SV; controls for local data quality |
| B14 | `mean_nm_rate` | Local alignment noise | Mean of `NM` ÷ aligned length over overlapping reads | float | 0–1 | NaN if no reads | High mismatch rate signals poor alignment, which generates spurious SV signatures |
| B15 | `n_supplementary` | Supplementary alignment count at the breakpoint | Count of reads with the supplementary flag set within ±`W_bp` | int | ≥0 | 0 | Excess supplementary alignments indicates multi-mapping / segmental duplication |

## Group A — Annotation-derived (reference context)

| # | Feature | Definition | Calculation | Type | Range | Missing | Biological meaning / why useful |
|---|---|---|---|---|---|---|---|
| A1 | `tr_overlap_frac` | **Required `repeat_context` (1/3).** Fraction of the SV span inside a tandem repeat | Overlap with UCSC `simpleRepeat` ÷ span length | float | 0–1 | 0 | Tandem repeats are the dominant source of long-read FP indels; also where many real SVs live |
| A2 | `rmsk_overlap_frac` | **Required `repeat_context` (2/3).** Fraction of span inside a RepeatMasker element | Overlap with UCSC `rmsk` ÷ span length | float | 0–1 | 0 | Alu/LINE elements drive real mobile-element insertions *and* mismapping |
| A3 | `segdup_overlap` | **Required `repeat_context` (3/3).** Overlap with segmental duplications | Overlap with `genomicSuperDups` ÷ span length | float | 0–1 | 0 | Segdups cause paralogous mismapping; a top source of false DUP/INV |
| A4 | `dist_to_nearest_gap` | Distance to the nearest assembly gap / N-run | `bedtools closest` against the gap track | int (bp) | ≥0 | large sentinel (1e9) | Calls adjacent to gaps are frequently assembly artefacts |
| A5 | `gc_content` | GC fraction of the span plus flanks | GC bases ÷ total, over POS−`W_flank` .. END+`W_flank` | float | 0–1 | NaN | Extreme GC correlates with coverage bias and basecalling error |
| A6 | `homopolymer_len_at_bp` | Longest homopolymer run at the breakpoint | Max run length within ±50 bp of POS | int | ≥1 | NaN | Homopolymers are the classic ONT error mode; short indel artefacts cluster here |
| A7 | `bp_microhomology_len` | Microhomology length at the junction | Longest common prefix/suffix between the two breakpoint flanks | int | ≥0 | NaN | Distinguishes repair-mechanism signatures of real SVs from alignment ambiguity |
| A8 | `chrom_class` | Chromosome category | autosome / chrX / chrY / other, one-hot | categorical | — | n/a | Sex chromosomes have different expected ploidy, changing depth-ratio interpretation. Raw chromosome identity is **excluded** — it is the split grouping variable |

## Group C — Caller-reported evidence (ablated)

| # | Feature | Definition | Calculation | Type | Range | Missing | Biological meaning / why useful |
|---|---|---|---|---|---|---|---|
| C1 | `sniffles_support` | **Required `breakpoint_support` (1/2).** Sniffles2 supporting-read count | `INFO/SUPPORT` | int | ≥0 | NaN + indicator `has_sniffles` | The caller's own evidence count; strong but not independent |
| C2 | `cutesv_support` | **Required `breakpoint_support` (2/2).** cuteSV supporting-read count | `INFO/RE` | int | ≥0 | NaN + indicator `has_cutesv` | Same signal from an independent implementation; disagreement is informative |
| C3 | `n_callers` | Caller agreement count | 1 or 2, from merge provenance | int | 1–2 | n/a | Encodes the intersection baseline. Must be present so the model can be shown to beat it, not just match it |
| C4 | `bp_offset_between_callers` | Breakpoint disagreement | \|POS_sniffles − POS_cutesv\| | int (bp) | ≥0 | NaN if `n_callers`==1 | Tight agreement indicates a well-defined real breakpoint |
| C5 | `svlen_disagreement` | Size disagreement | \|SVLEN_sniffles − SVLEN_cutesv\| | int (bp) | ≥0 | NaN if `n_callers`==1 | Large disagreement flags an ambiguous or repeat-driven event |
| C6 | `caller_qual` | Reported quality | VCF `QUAL`; mean if both callers | float | ≥0 | NaN | Caller's own confidence, kept for comparison against the learned score |
| C7 | `filter_pass` | Caller FILTER status | 1 if PASS in ≥1 caller | binary | 0/1 | 0 | Encodes the callers' own built-in heuristic filter |

## Group V — Caller-declared type and size (ablated; leakage-sensitive)

| # | Feature | Definition | Calculation | Type | Range | Missing | Biological meaning / why useful |
|---|---|---|---|---|---|---|---|
| V1 | `log10_svlen` | Log-scaled SV length | log10(\|SVLEN\| + 1) | float | ≥0 | NaN for BND | SV length distribution is strongly type- and artefact-dependent (Alu ~300 bp, LINE ~6 kb peaks) |
| V2 | `svtype_called` | Caller-declared type | one-hot DEL/INS/DUP/INV/BND | categorical | — | n/a | **Leakage risk for Stage B** — see below |
| V3 | `span_len` | Reference interval length | END − POS | int | ≥0 | 0 for INS | Distinguishes reference-consuming events from insertions; pairs with B2 |

### Leakage control on V2

`svtype_called` and the **sign** of `SVLEN` both directly encode the Stage B target
(DEL vs INS). Therefore:

- **Stage A (validity):** V2 is allowed. Predicting validity from declared type is not
  circular.
- **Stage B (DEL vs INS):** V2 is **excluded**, and `log10_svlen` uses the **absolute**
  value so the sign cannot leak the answer. Reported both ways, with the V2-included run
  labelled explicitly as the leaky upper bound for reference only.

---

## Missing-value strategy

Three distinct kinds of missingness, handled differently rather than uniformly imputed:

1. **Structurally undefined** (`dp_flank_ratio` for an insertion; `bp_offset_between_callers`
   for a single-caller candidate). Encode as `NaN` **plus a boolean indicator column**.
   XGBoost learns a default split direction for NaN natively; Random Forest receives a
   sentinel (−1) plus the indicator, since scikit-learn's forest does not accept NaN.
2. **Genuinely zero** (`n_split_reads` where no split reads exist). Fill with 0 — this is
   a real measurement, not missing data.
3. **Uncomputable** (no reads at the locus). Fill with `NaN` + indicator, and log the
   candidate. If this affects more than ~1% of candidates it indicates a pipeline bug and
   is escalated rather than imputed away.

Imputation parameters are fit on the **training chromosomes only** and applied unchanged to
validation and test, inside a scikit-learn `Pipeline`, so no test-set statistics leak into
preprocessing.

---

## Validation gates before a feature is confirmed (Phase 6)

Each feature must pass all of these, with results recorded in
`notebooks/02_feature_sanity.ipynb`:

1. **Non-degenerate** — not constant; not >95% a single value.
2. **Unit-tested** — computed correctly on a hand-checked fixture in `tests/data/`.
3. **Distribution inspected** — plotted, split by `is_true`, and visually sane.
4. **Direction sanity-checked** — sign of the relationship matches the stated biology
   (e.g. `dp_flank_ratio` must be lower for true DELs than for false ones). A feature whose
   real behaviour contradicts its stated biological meaning gets its rationale corrected or
   is dropped; it is not kept with a wrong explanation attached.
5. **Cost-checked** — extraction time per candidate is feasible at the Tier-2 scale.
6. **Not a truth proxy** — cannot be computed from the truth VCF, directly or indirectly.
