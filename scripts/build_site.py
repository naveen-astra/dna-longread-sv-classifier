"""Build the static demo site (site/index.html) from a finished pipeline run.

Scores every labelled candidate in the held-out test chromosome with the trained Stage A
Random Forest, computes per-candidate SHAP reasons, and inlines the result into
scripts/site_template.html. The output is a single self-contained HTML file: it opens
from disk with no server and can be hosted as-is (e.g. GitHub Pages from /site).

Run inside the svml environment after the pipeline has finished:

    python scripts/build_site.py --results ~/svml-work/results
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from svml.ml.train import _prepare, load_model

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "scripts" / "site_template.html"
TEST_CHROM = "chr22"
SHOWN_FEATURES = [
    "n_callers", "sniffles_support", "cutesv_support", "caller_qual", "filter_pass",
    "svlen_disagreement", "bp_offset_between_callers", "support_ratio", "mean_mapq",
    "dp_flank_ratio", "n_split_reads", "tr_overlap_frac", "segdup_overlap", "gc_content",
]
TOP_REASONS = 7


def _num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 4)


def build_payload(results: Path) -> dict:
    import shap

    model, encoder, meta = load_model(results / "models" / "stageA_random_forest_B+A+C+V.pkl")
    threshold = float(meta.threshold)
    forest = model.calibrated_classifiers_[0].estimator

    dataset = pd.read_parquet(results / "ml" / "dataset.parquet")
    test = dataset[(dataset["chrom"] == TEST_CHROM) & dataset["is_true"].notna()]
    test = test.reset_index(drop=True)
    x = _prepare(encoder, test, "random_forest")
    p_calibrated = model.predict_proba(x)[:, 1]
    p_forest = forest.predict_proba(x)[:, 1]

    explainer = shap.TreeExplainer(forest)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[1]
    elif np.ndim(values) == 3:
        values = values[:, :, 1]
    base = explainer.expected_value
    base = float(base[1] if np.ndim(base) else base)

    rows = []
    for i, rec in test.iterrows():
        contrib = pd.Series(values[i], index=x.columns)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(TOP_REASONS)
        rows.append({
            "id": str(rec["candidate_id"]),
            "pos": int(rec["pos"]),
            "end": int(rec["end"]) if pd.notna(rec.get("end")) else None,
            "type": str(rec["svtype_called"]),
            "len": int(rec["svlen"]),
            "p": round(float(p_calibrated[i]), 4),
            "pf": round(float(p_forest[i]), 4),
            "truth": int(float(rec["is_true"])),
            "f": {k: _num(rec[k]) for k in SHOWN_FEATURES if k in test.columns},
            "why": [[k, round(float(v), 4), _num(x.iloc[i][k])] for k, v in top.items()],
        })

    bench = json.loads((results / "benchmark" / "stageA_metrics.json").read_text())
    ops = {k: {"p": round(v["precision"], 4), "r": round(v["recall"], 4), "f1": round(v["f1"], 4)}
           for k, v in bench["view1_operating_points"].items()}
    curves = {k: {"p": [round(a, 4) for a in v["precision"]], "r": [round(a, 4) for a in v["recall"]]}
              for k, v in bench["view3_pr_curves"].items()}
    return {
        "chrom": TEST_CHROM,
        "threshold": threshold,
        "base": round(base, 4),
        "ceiling": round(bench["recall_ceiling_union"], 4),
        "missed": bench["truth_svs_missed_by_both_callers"],
        "truth_total": bench["truth_positives_total"],
        "matched_gain": bench["contribution"]["rf"]["delta_precision_at_matched_recall"],
        "ops": ops,
        "pr": curves,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("~/svml-work/results"),
                        help="pipeline results directory (default: ~/svml-work/results)")
    parser.add_argument("--out", type=Path, default=REPO / "site" / "index.html")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    payload = build_payload(args.results.expanduser())
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8", newline="\n")
    rows = payload["rows"]
    correct = sum((r["p"] >= payload["threshold"]) == (r["truth"] == 1) for r in rows)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB): "
          f"{len(rows)} candidates, {correct / len(rows):.1%} correct at {payload['threshold']:.2f}")


if __name__ == "__main__":
    main()
