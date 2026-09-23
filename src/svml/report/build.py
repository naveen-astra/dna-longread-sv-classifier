"""Report assembly (spec section 19: all 20 sections; outputs P and Q).

Two hard rules enforced here:

  * Nothing illustrative ever enters a results table (spec section 28). Absent values
    render as "TBD", never as a plausible-looking number.
  * Every stage C number carries its pseudo-label provenance in the same table as the
    number, not in a footnote (finding B1).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment

from svml.utils.logging import get_logger

log = get_logger(__name__)

PLACEHOLDER = "TBD"

STAGE_C_WARNING = (
    "Pseudo-label derived; NOT GIAB-benchmarked. No GIAB truth set labels DUP, INV or "
    "TRA, so these numbers measure agreement with the callers' consensus, which is a "
    "much weaker quantity than accuracy."
)

LIMITATIONS = [
    "No DUP / INV / TRA ground truth exists in GIAB. The NIST_SV_v0.6 benchmark is "
    "GRCh37 and DEL/INS only; the GRCh38-native v5.0q stvar benchmark excludes "
    "inversion interiors from its confident regions. Five-class results rest on "
    "caller-consensus pseudo-labels and are not benchmarked.",
    "Only one labelled sample (HG002) has SV truth, so generalisation is tested across "
    "sequencing platforms rather than across samples.",
    "The analysis is region-scoped rather than whole-genome, to stay within the "
    "compute and storage budget.",
    "Benchmark confident regions systematically exclude many of the hardest genomic "
    "regions, so the project measures a repeat-context gradient but cannot benchmark "
    "the most repetitive loci. No benchmark can.",
    "GIAB v5.0q is a draft benchmark that does not exclude complex SVs; some apparent "
    "false positives are representation differences or benchmark errors rather than "
    "model errors.",
    "The ML layer is a filter over the union of the two callers' candidates. Its recall "
    "is hard-capped by that union and it cannot discover SVs neither caller proposed.",
    "This is a research-grade prototype. It is not clinically validated and makes no "
    "medical claims.",
]

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  :root { --fg:#1a1a1a; --muted:#666; --line:#e2e4e8; --accent:#2c5aa0;
          --warn-bg:#fff8e6; --warn-br:#e0a800; --bad-bg:#fdecea; --bad-br:#c0392b; }
  body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         line-height:1.6; color:var(--fg); max-width:1100px; margin:0 auto; padding:2.5rem 1.5rem; }
  h1 { border-bottom:3px solid var(--accent); padding-bottom:.4rem; }
  h2 { margin-top:2.4rem; color:var(--accent); border-bottom:1px solid var(--line); padding-bottom:.3rem; }
  h3 { margin-top:1.6rem; font-size:1.05rem; }
  table { border-collapse:collapse; width:100%; margin:1rem 0; font-size:.9rem; }
  th,td { border:1px solid var(--line); padding:.5rem .7rem; text-align:left; }
  th { background:#f5f7fa; font-weight:600; }
  tr:nth-child(even) td { background:#fafbfc; }
  .meta { color:var(--muted); font-size:.88rem; }
  .warn { background:var(--warn-bg); border-left:4px solid var(--warn-br); padding:.9rem 1.1rem; margin:1.1rem 0; }
  .bad  { background:var(--bad-bg);  border-left:4px solid var(--bad-br);  padding:.9rem 1.1rem; margin:1.1rem 0; }
  .tbd { color:var(--muted); font-style:italic; }
  img { max-width:100%; border:1px solid var(--line); border-radius:4px; margin:1rem 0; }
  code { background:#f2f4f7; padding:.1rem .35rem; border-radius:3px; font-size:.87em; }
  ol.limitations li { margin-bottom:.55rem; }
</style>
</head>
<body>

<h1>{{ title }}</h1>
<p class="meta">
  Generated {{ generated }} &middot; config digest <code>{{ config_digest }}</code>
  &middot; seed {{ seed }} &middot; svml {{ version }}
</p>

<div class="warn">
<strong>Scope of claims.</strong> The ML layer is a filter over the union of Sniffles2 and
cuteSV candidates. Its recall is hard-capped at the union's recall
({{ recall_ceiling }}); it cannot discover SVs neither caller proposed. Comparisons are
reported against the <strong>intersection</strong> baseline &mdash; the trivial
"both callers agree" filter &mdash; because beating a single caller proves nothing.
</div>

<h2>1&ndash;3. Dataset, platform and reference</h2>
<table>
<tr><th>Sample</th><th>Platform</th><th>Role</th><th>Truth set</th><th>Coverage</th></tr>
{% for s in samples %}
<tr><td>{{ s.sample_id }}</td><td>{{ s.platform }}</td><td>{{ s.role }}</td>
    <td>{{ s.truth }}</td><td>{{ s.coverage | tbd }}</td></tr>
{% endfor %}
</table>
<p>Reference: <code>{{ reference.id | tbd }}</code> ({{ reference.build | tbd }}).
   Truth: {{ truth.release | tbd }} on {{ truth.build | tbd }}.</p>

<h2>4. QC results</h2>
{{ qc_table }}
{% if figures.qc %}<img src="{{ figures.qc }}" alt="QC summary">{% endif %}

<h2>5. Alignment statistics</h2>
{{ align_table }}

<h2>6&ndash;7. SV caller statistics and candidate counts</h2>
{{ candidate_table }}
{% if figures.sv_type %}<img src="{{ figures.sv_type }}" alt="SV type distribution">{% endif %}
{% if figures.sv_size %}<img src="{{ figures.sv_size }}" alt="SV size distribution">{% endif %}
{% if figures.karyotype %}<img src="{{ figures.karyotype }}" alt="Karyotype view">{% endif %}

<h3>Confident-region gate</h3>
<p>A candidate outside the benchmark's confident regions is <em>unknown</em>, not false.
Labelling those as negatives would teach the model that real SVs in hard regions are
artefacts. Counts:</p>
{{ gate_table }}

<h2>8. Feature distributions</h2>
<p>{{ feature_count }} features across four ablation groups
(B = BAM-derived, A = annotation-derived, C = caller-reported, V = caller-declared).</p>
{% if figures.features %}<img src="{{ figures.features }}" alt="Feature distributions">{% endif %}

<h2>9. ML methodology</h2>
<p>Two-stage design (docs/04_ml_task_definition.md). Split strategy:
<strong>{{ split.strategy }}</strong> &mdash; {{ split.rationale }}</p>
{{ split_table }}

<h2>10&ndash;15. Model comparison and results</h2>
<h3>View 1 &mdash; operating points</h3>
{{ results_table }}
<h3>View 2 &mdash; matched comparison</h3>
{{ matched_table }}
<h3>View 3 &mdash; precision&ndash;recall curves</h3>
{% if figures.pr %}<img src="{{ figures.pr }}" alt="PR curves">{% endif %}
{% for name, path in figures.confusion.items() %}
<h3>Confusion matrix &mdash; {{ name }}</h3>
<img src="{{ path }}" alt="Confusion matrix {{ name }}">
{% endfor %}

<h3>Ablation</h3>
<p>The <code>B+A</code> model uses only evidence computed independently of the callers.
If performance collapses without groups C and V, the model is largely recalibrating the
callers' own confidence rather than contributing independent evidence.</p>
{{ ablation_table }}

<h3>Stage C &mdash; five-class</h3>
<div class="warn"><strong>{{ stage_c_warning }}</strong></div>
{{ stage_c_table }}

<h2>16. SHAP interpretation</h2>
{{ shap_table }}
{% for path in figures.shap %}<img src="{{ path }}" alt="SHAP plot">{% endfor %}

<h2>17. Sniffles2 vs cuteSV vs ML</h2>
{{ contribution_table }}
{% if not claim_supported %}
<div class="bad"><strong>The precision claim is not supported by this run.</strong>
No ML model exceeded the {{ headline_reference }} baseline's precision at matched
recall. Reported as measured.</div>
{% endif %}

<h3>Region claim</h3>
<p>{{ region_claim.note | tbd }}</p>
{% if figures.strata %}<img src="{{ figures.strata }}" alt="Precision by repeat context">{% endif %}
<p class="meta">{{ region_claim.coverage_caveat | tbd }}</p>

<h2>18. Error analysis</h2>
{{ error_table }}

<h2>19. Limitations</h2>
<ol class="limitations">
{% for item in limitations %}<li>{{ item }}</li>{% endfor %}
</ol>

<h2>20. Conclusion</h2>
<p>{{ conclusion }}</p>

<h2>Reproducibility</h2>
{{ provenance_table }}

</body>
</html>
"""


def _tbd(value: Any) -> str:
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return f'<span class="tbd">{PLACEHOLDER}</span>'
    return str(value)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return f'<span class="tbd">{PLACEHOLDER}</span>'
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]], empty_note: str | None = None) -> str:
    if not rows:
        note = empty_note or f"No data available ({PLACEHOLDER})."
        return f'<p class="tbd">{note}</p>'
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _ci(result: dict[str, Any], key: str) -> str:
    interval = (result.get("ci") or {}).get(key)
    if not interval:
        return PLACEHOLDER
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def build_report(
    out_path: str | Path,
    metadata: dict[str, Any],
    benchmark: dict[str, Any] | None = None,
    candidate_summary: dict[str, Any] | None = None,
    gate_counts: dict[str, Any] | None = None,
    qc_metrics: dict[str, Any] | None = None,
    align_stats: dict[str, Any] | None = None,
    figures: dict[str, Any] | None = None,
    shap_summaries: list[dict[str, Any]] | None = None,
    ablation: dict[str, Any] | None = None,
    stage_c: dict[str, Any] | None = None,
    error_analysis: dict[str, Any] | None = None,
    split_info: dict[str, Any] | None = None,
    feature_count: int = 0,
) -> Path:
    """Render the HTML report. Missing sections render as TBD, never as invented numbers."""
    benchmark = benchmark or {}
    figures = figures or {}
    figures.setdefault("confusion", {})
    figures.setdefault("shap", [])

    view1 = benchmark.get("view1_operating_points", {})
    results_rows = [
        [
            name,
            r.get("precision"),
            _ci(r, "precision"),
            r.get("recall"),
            _ci(r, "recall"),
            r.get("f1"),
            r.get("tp"),
            r.get("fp"),
            r.get("fn"),
        ]
        for name, r in view1.items()
    ]

    matched_rows = [
        [
            name,
            m.get("reference"),
            m.get("at_reference_recall", {}).get("precision"),
            m.get("reference_precision"),
            m.get("at_reference_precision", {}).get("recall"),
            m.get("reference_recall"),
        ]
        for name, m in benchmark.get("view2_matched", {}).items()
    ]

    contribution_rows = [
        [
            name,
            c.get("reference"),
            c.get("delta_f1_vs_reference"),
            c.get("delta_precision_at_matched_recall"),
            "yes" if c.get("claim_supported") else "no",
        ]
        for name, c in benchmark.get("contribution", {}).items()
    ]

    shap_rows = []
    for summary in shap_summaries or []:
        for entry in summary.get("consistency", {}).get("top_features", [])[:10]:
            shap_rows.append(
                [
                    f"{summary.get('stage')}/{summary.get('model')}",
                    entry["feature"],
                    entry["group"],
                    entry["mean_abs_shap"],
                    entry["stated_meaning"][:110],
                ]
            )

    samples = metadata.get("samples", [])
    split_info = split_info or {}
    region_claim = (benchmark.get("region_claim") or {}) if benchmark else {}

    # The filter must be registered on the environment BEFORE the template is
    # compiled; Jinja resolves filter names at compile time, not at render time.
    env = Environment(autoescape=False)  # noqa: S701 - trusted local template
    env.filters["tbd"] = _tbd
    template = env.from_string(_TEMPLATE)

    html = template.render(
        title="AI-Based Structural Variant Detection Using Long-Read Sequencing Data",
        generated=metadata.get("generated_utc", datetime.now(UTC).isoformat()),
        config_digest=metadata.get("config_digest", PLACEHOLDER),
        seed=metadata.get("random_seed", PLACEHOLDER),
        version=metadata.get("svml_version", PLACEHOLDER),
        recall_ceiling=_fmt(benchmark.get("recall_ceiling_union")),
        headline_reference=benchmark.get("headline_reference", "intersection"),
        samples=samples,
        reference=metadata.get("reference", {}),
        truth=metadata.get("truth", {}),
        qc_table=_table(
            ["Sample", "Reads", "Total bases", "N50", "Median Q"],
            [
                [s, m.get("n_reads"), m.get("total_bases"), m.get("read_n50"), m.get("median_quality")]
                for s, m in (qc_metrics or {}).items()
            ],
        ),
        align_table=_table(
            ["Sample", "Mapped reads", "Mapping rate", "Mean depth"],
            [
                [s, m.get("mapped_reads"), m.get("mapping_rate"), m.get("mean_depth")]
                for s, m in (align_stats or {}).items()
            ],
        ),
        candidate_table=_table(
            ["Metric", "Value"],
            [[k, v] for k, v in (candidate_summary or {}).items()],
        ),
        gate_table=_table(
            ["Bucket", "Count"],
            [[k, v] for k, v in (gate_counts or {}).items()],
        ),
        split_table=_table(
            ["Split", "Chromosomes"],
            [[k, ", ".join(v)] for k, v in (metadata.get("split_map_by_split") or {}).items()],
        ),
        split=split_info or {"strategy": PLACEHOLDER, "rationale": PLACEHOLDER},
        results_table=_table(
            ["Baseline", "Precision", "P 95% CI", "Recall", "R 95% CI", "F1", "TP", "FP", "FN"],
            results_rows,
            "Benchmark has not been run yet.",
        ),
        matched_table=_table(
            ["Model", "Reference", "Precision @ ref recall", "Reference precision",
             "Recall @ ref precision", "Reference recall"],
            matched_rows,
            "Matched comparison unavailable until the benchmark has run.",
        ),
        ablation_table=_table(
            ["Stage", "Model", "Ablation", "Macro-F1", "Precision", "Recall"],
            [
                [a.get("stage"), a.get("model"), a.get("ablation"),
                 a.get("macro_f1"), a.get("precision"), a.get("recall")]
                for a in (ablation or {}).get("rows", [])
            ],
        ),
        stage_c_table=_table(
            ["Model", "Class", "Precision", "Recall", "F1", "Support", "Label provenance"],
            [
                [row.get("model"), row.get("label"), row.get("precision"), row.get("recall"),
                 row.get("f1"), row.get("support"), "caller-consensus pseudo-label"]
                for row in (stage_c or {}).get("rows", [])
            ],
            "Stage C has not been run yet.",
        ),
        stage_c_warning=STAGE_C_WARNING,
        shap_table=_table(
            ["Model", "Feature", "Group", "Mean |SHAP|", "Stated biological meaning"],
            shap_rows,
            "SHAP has not been run yet.",
        ),
        contribution_table=_table(
            ["Model", "Reference", "ΔF1 vs reference", "ΔPrecision @ matched recall", "Claim supported"],
            contribution_rows,
        ),
        claim_supported=bool(benchmark.get("claim_supported_by")),
        region_claim=region_claim,
        error_table=_table(
            ["Category", "Count", "Notes"],
            [
                [k, v.get("count"), v.get("note", "")]
                for k, v in (error_analysis or {}).items()
                if isinstance(v, dict)
            ],
            "Error analysis has not been run yet. It is required, not optional "
            "(docs/05_evaluation_protocol.md).",
        ),
        provenance_table=_table(
            ["Tool", "Version"],
            [[k, v or PLACEHOLDER] for k, v in (metadata.get("tool_versions") or {}).items()],
        ),
        figures=figures,
        feature_count=feature_count,
        limitations=LIMITATIONS,
        conclusion=_conclusion(benchmark),
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("wrote report to %s", out)
    return out


def _conclusion(benchmark: dict[str, Any]) -> str:
    if not benchmark or not benchmark.get("view1_operating_points"):
        return (
            f"{PLACEHOLDER} &mdash; the benchmark has not been run, so no conclusion is "
            "stated. Nothing illustrative is presented as a result (spec section 28)."
        )
    supported = benchmark.get("claim_supported_by") or []
    reference = benchmark.get("headline_reference", "intersection")
    if supported:
        return (
            f"At matched recall, {', '.join(supported)} exceeded the {reference} baseline's "
            "precision, so the precision claim is supported for those models on this test "
            "set. See the region-claim section for whether the gain concentrates in "
            "repeat-rich regions, and the limitations for what this does not establish."
        )
    return (
        f"No ML model exceeded the {reference} baseline's precision at matched recall on "
        "this test set, so the precision claim is not supported by this run. This is "
        "reported as measured rather than re-tuned against the test data."
    )


def write_summary_json(out_path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("wrote machine-readable summary to %s", path)
    return path


def html_to_pdf(html_path: str | Path, pdf_path: str | Path) -> Path | None:
    """Render the HTML report to PDF. Returns None if WeasyPrint is unavailable."""
    try:
        from weasyprint import HTML
    except ImportError:
        log.warning("WeasyPrint not installed; skipping PDF output")
        return None

    pdf = Path(pdf_path)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        HTML(filename=str(html_path)).write_pdf(str(pdf))
    except Exception as exc:  # noqa: BLE001 - PDF is a nice-to-have, not a blocker
        log.warning("PDF rendering failed: %s", exc)
        return None
    log.info("wrote PDF report to %s", pdf)
    return pdf
