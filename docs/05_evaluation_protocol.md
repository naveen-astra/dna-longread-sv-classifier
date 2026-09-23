# Evaluation Protocol

Required by PROJECT MASTER CONTEXT §26(G). Implements findings H2 (fair comparison), H5
(standard tools) and L1 (stratification).

---

## The comparison problem

The ML layer is a **filter over the union of Sniffles2 and cuteSV candidates**. It only
ever removes candidates; it cannot discover an SV neither caller proposed. Therefore:

- **ML recall is hard-capped at the union's recall.** This ceiling is a property of the
  architecture, not a result, and must be printed next to every ML recall number.
- ML will essentially always beat the union on precision and lose on recall.
- A single operating point can be chosen to make the ML layer look arbitrarily good.

Comparing a filter to a caller at one threshold is not a meaningful comparison. The
protocol below removes that degree of freedom.

---

## Baselines (all six, same truth, same matcher)

| # | Baseline | Definition | Role |
|---|---|---|---|
| 1 | Sniffles2 | Sniffles2 PASS calls alone | Published heuristic caller |
| 2 | cuteSV | cuteSV PASS calls alone | Published heuristic caller |
| 3 | **Union** | All merged candidates | **Recall ceiling** for every ML model |
| 4 | **Intersection** | Candidates called by both (`n_callers == 2`) | **The trivial heuristic filter the ML layer must beat** |
| 5 | RF | Random Forest Stage A filter | Contribution |
| 6 | XGBoost | XGBoost Stage A filter | Contribution |

Baselines 3 and 4 are the additions from finding H2, and 4 is the one that matters. A
reader's first objection to "ML filtered the candidates and precision went up" is *"so does
requiring both callers to agree."* If the ML layer does not beat the intersection, it has
not demonstrated anything, and the report must say so plainly.

Every baseline is evaluated with **identical** truth, confident regions, and matching
parameters. No baseline gets a tuned variant that the others do not.

---

## Matching and metrics

**Tool: `truvari bench`.** Used for both label creation and evaluation, with the same
parameters, so the labeller and evaluator cannot disagree about what a match is (H5). For
the v5.0q draft benchmark, `--refine` is used as that benchmark's README recommends.

Parameters pinned in `config.yaml`:

```yaml
truvari:
  refdist: 500        # max breakpoint distance
  pctseq: 0.7         # min sequence similarity
  pctsize: 0.7        # min size similarity
  refine: true        # recommended for v5.0q
  passonly: true
```

Reported per baseline:

- Precision, Recall, F1
- Per-class precision / recall / F1 (DEL, INS)
- Macro-F1
- Confusion matrix
- TP / FP / FN counts (absolute, not only rates)
- False-positive rate and false-negative count (§15)

Plus, for RF and XGBoost only:

- Full **precision–recall curve** over the score threshold
- Average precision (area under PR)
- **Calibration curve** and Brier score, since §10 requires confidence outputs and an
  uncalibrated probability is not a confidence

---

## Required comparison views

A single table of one-point-per-model is not sufficient. Three views:

**View 1 — Operating-point table.** Each baseline at its natural operating point, with the
union recall ceiling stated in the caption. RF/XGBoost at the threshold selected on the
**validation** split.

**View 2 — Matched comparison.** The one that supports the actual claim:

- *At the intersection baseline's recall*, what is each ML model's precision?
- *At the intersection baseline's precision*, what is each ML model's recall?

If ML precision at matched recall does not exceed the intersection's precision, the
precision claim is not supported. State that outcome if it occurs.

**View 3 — PR curve overlay.** RF and XGBoost curves, with baselines 1–4 plotted as single
points. This makes the whole tradeoff visible and is the honest headline figure.

**Headline contribution metric:** ΔF1 and Δprecision-at-matched-recall, both measured
against **baseline 4 (intersection)**, not against a single caller.

---

## Stratification (finding L1)

The region claim is only testable if metrics are broken out by difficulty. Every model in
View 1 is additionally reported across:

| Stratum | Definition |
|---|---|
| Repeat context | `tr_overlap_frac` in [0, 0.1), [0.1, 0.5), [0.5, 1.0] |
| Segmental duplication | `segdup_overlap` == 0 vs > 0 |
| SV size | 50–100 bp, 100 bp–1 kb, 1–10 kb, >10 kb |
| SV type | DEL, INS |
| Local MAPQ | `frac_mapq0` == 0 vs > 0 |

**The region claim is supported only if the precision gain over baseline 4 is larger in the
high-repeat stratum than in the low-repeat stratum.** That specific comparison is the
project's central empirical result and gets its own figure.

### Coverage caveat that must appear in Limitations

Benchmark confident regions systematically exclude many of the hardest parts of the genome
— v0.6 excluded large tandem repeats and segdups, and v5.0q excludes inversion interiors.
So the "high-repeat" stratum means *the most repetitive regions for which truth exists*,
not the most repetitive regions in the genome. The project can measure a repeat-context
gradient honestly; it cannot benchmark the genuinely hardest regions, because nobody can.
Report the fraction of candidates falling outside confident regions as a first-class number.

---

## Ablation (finding M3)

Both feature sets, both models, all reported:

| Model | Features | Question answered |
|---|---|---|
| RF / XGB `B+A` | BAM + annotation only | Does **independent** evidence add anything? |
| RF / XGB `B+A+C+V` | All features | Best achievable performance |

If performance collapses without Groups C and V, the model is largely recalibrating the
callers' own confidence rather than contributing independent evidence. That is a legitimate
and publishable finding, but it changes the conclusion, so it must be reported either way
rather than only when favourable.

---

## Statistical reporting

- **Confidence intervals** on precision/recall/F1 via bootstrap over candidates
  (1000 resamples), reported for every headline number. A difference without an interval is
  not a result.
- **Model comparison** (RF vs XGBoost): paired bootstrap on the test split. Do not declare a
  winner if the intervals overlap — §28 forbids claiming superiority before comparison
  demonstrates it.
- **Seeds fixed and recorded** for every stochastic step (§23).

---

## Error analysis (report §19 item 18)

Required, not optional:

1. Sample **50 false positives** and **50 false negatives** from the test split.
2. Cross-tabulate against strata to identify where the model fails systematically.
3. Manually curate a subset in IGV. The v5.0q README specifically recommends curating
   putative FPs/FNs, because some apparent errors are representation differences rather
   than real errors, and in a draft benchmark some are benchmark errors.
4. Report how many "errors" were genuine model errors, representation differences, or
   likely benchmark errors.

Step 4 is what separates a defensible evaluation from a number dumped out of a tool.

---

## Execution discipline

1. Test chromosomes are **not read** until every model and threshold is frozen.
2. `results/benchmark/metrics.json` is written once per frozen configuration and the
   config hash is recorded alongside it.
3. Any change after seeing test results requires re-freezing and a documented re-run, and
   the fact that it happened is recorded in the report.
4. Nothing illustrative ever enters a results table (§28). Placeholder values in draft
   documents are written as `TBD`, never as plausible-looking numbers.
