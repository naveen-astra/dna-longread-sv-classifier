"""Build the static project site (site/index.html) from a finished pipeline run.

Collects the project facts (samples, splits, features, models, benchmark results) and the
per-candidate predictions of the trained Stage A Random Forest on the held-out test
chromosome, then inlines them into scripts/site_template.html. The output is a single
self-contained HTML file: it opens from disk with no server and can be hosted as-is
(e.g. GitHub Pages from /site).

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

from svml.features.registry import FEATURES, FEATURES_BY_NAME
from svml.ml.train import _prepare, load_model

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "scripts" / "site_template.html"
TEST_CHROM = "chr22"
SHOWN_FEATURES = [
    "n_callers", "sniffles_support", "cutesv_support", "caller_qual", "filter_pass",
    "svlen_disagreement", "bp_offset_between_callers", "support_ratio", "mean_mapq",
    "dp_flank_ratio", "n_split_reads", "tr_overlap_frac", "segdup_overlap", "gc_content",
]
# Plain-language wording for the site where the registry definition uses code terms
# (parameter names, VCF fields). The registry itself stays technical for the docs.
PLAIN_DEFINITIONS = {
    "dp_local": "Mean read depth around the breakpoint.",
    "n_split_reads": "Reads split into pieces that align on either side of the breakpoint.",
    "n_clipped_reads": "Reads with clipped ends near the breakpoint.",
    "clip_frac": "Share of local reads that are clipped.",
    "n_spanning_reads": "Reads crossing the whole SV without a large insertion or deletion.",
    "strand_balance": "How unevenly the supporting reads come from the two DNA strands.",
    "mean_nm_rate": "Average mismatch rate of reads overlapping the breakpoint.",
    "gc_content": "GC fraction across the SV and its flanking sequence.",
    "sniffles_support": "Reads Sniffles2 reports as supporting the SV.",
    "cutesv_support": "Reads cuteSV reports as supporting the SV.",
    "caller_qual": "Caller quality score, averaged when both callers report the SV.",
    "log10_svlen": "SV length on a log scale.",
    "span_len": "Length of reference sequence the SV covers. Zero for insertions.",
}
TOP_REASONS = 7
TOP_IMPORTANCE = 8
BOOTSTRAPS = 2000
SEED = 42


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _num(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 4)


def _best_precision(score: np.ndarray, y: np.ndarray, total_true: float, target: float):
    """Best precision among thresholds whose recall (over all truth SVs) reaches `target`."""
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    tp, fp = np.cumsum(yy == 1), np.cumsum(yy == 0)
    at_threshold = np.append(s[1:] != s[:-1], True)   # only cut between distinct scores
    ok = at_threshold & (tp / total_true >= target - 1e-9)
    return float((tp / (tp + fp))[ok].max()) if ok.any() else None


def _gain_interval(y, model_score, intersection, missed):
    """Bootstrap interval for (model precision - intersection precision) at equal recall."""
    rng = np.random.default_rng(SEED)
    gains = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(y), len(y))
        yb, ib = y[idx], intersection[idx]
        total = (yb == 1).sum() + missed
        tp, fp = ((ib == 1) & (yb == 1)).sum(), ((ib == 1) & (yb == 0)).sum()
        if tp + fp == 0:
            continue
        best = _best_precision(model_score[idx], yb, total, tp / total)
        if best is not None:
            gains.append(best - tp / (tp + fp))
    g = np.array(gains)
    return (float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)), float((g <= 0).mean()))


def _samples(results: Path, summary: dict) -> list[dict]:
    out = []
    for s in summary["metadata"]["samples"]:
        sid = s["sample_id"]
        qc = _read(results / "qc" / sid / "qc_metrics.json")
        align = _read(results / "align" / f"{sid}.align_stats.json")
        merged = _read(results / "candidates" / f"{sid}.merge_summary.json")
        out.append({
            "id": sid, "platform": s["platform"], "role": s["role"],
            "reads": int(qc["n_reads"]), "mapped": align["mapping_rate"],
            "candidates": merged["n_candidates"],
        })
    return out


def _models(results: Path) -> list[dict]:
    rows = []
    for path in sorted((results / "models").glob("stageA_*.json")):
        m = _read(path)
        rows.append({
            "stage": m["stage"], "model": m["model_name"], "ablation": m["ablation"],
            "n_features": len(m["columns"]), "val": round(m["val_score"], 3),
            "threshold": m.get("threshold"),
        })
    return rows


def _ablation(summary: dict) -> list[dict]:
    keep = {"rf": "Random Forest", "xgboost": "XGBoost"}
    out = []
    for row in summary["ablation"]:
        if row["stage"] == "A" and row["model"] in keep:
            out.append({
                "model": keep[row["model"]],
                "all_features": row["ablation"].startswith("B+A+C+V"),
                "precision": round(row["precision"], 3), "recall": round(row["recall"], 3),
            })
    return out


def build_payload(results: Path) -> dict:
    import shap

    summary = _read(results / "report" / "summary.json")
    bench = _read(results / "benchmark" / "stageA_metrics.json")
    stage_b = _read(results / "benchmark" / "stageB_metrics.json")
    stage_c = _read(results / "benchmark" / "stageC_metrics.json")

    model, encoder, meta = load_model(results / "models" / "stageA_random_forest_B+A+C+V.pkl")
    forest = model.calibrated_classifiers_[0].estimator

    dataset = pd.read_parquet(results / "ml" / "dataset.parquet")
    test = dataset[(dataset["chrom"] == TEST_CHROM) & dataset["is_true"].notna()]
    test = test.reset_index(drop=True)
    x = _prepare(encoder, test, "random_forest")
    score = model.predict_proba(x)[:, 1]
    y = test["is_true"].astype(float).to_numpy()

    explainer = shap.TreeExplainer(forest)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[1]
    elif np.ndim(values) == 3:
        values = values[:, :, 1]

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
            "p": round(float(score[i]), 4),
            "truth": int(y[i]),
            "f": {k: _num(rec[k]) for k in SHOWN_FEATURES if k in test.columns},
            "why": [[k, round(float(v), 4), _num(x.iloc[i][k])] for k, v in top.items()],
        })

    importance = pd.Series(np.abs(values).mean(axis=0), index=x.columns)
    importance = importance.groupby(lambda c: c.split("__")[0]).sum()
    importance = importance.sort_values(ascending=False).head(TOP_IMPORTANCE)
    importance = [{"name": k, "value": round(float(v), 4),
                   "group": FEATURES_BY_NAME[k].group.value if k in FEATURES_BY_NAME else None}
                  for k, v in importance.items()]

    ops = {k: {"p": round(v["precision"], 6), "r": round(v["recall"], 6), "f1": round(v["f1"], 6),
               "ci": [round(c, 6) for c in v["ci"]["precision"]]}
           for k, v in bench["view1_operating_points"].items()}
    curves = {k: {"p": [round(a, 4) for a in v["precision"]], "r": [round(a, 4) for a in v["recall"]]}
              for k, v in bench["view3_pr_curves"].items()}

    # Checks that go beyond the benchmark file: is the gain over the heuristic distinguishable
    # from noise, and does a single caller-support rule do as well?
    total_true, missed = bench["truth_positives_total"], bench["truth_svs_missed_by_both_callers"]
    ref_recall = ops["intersection"]["r"]
    low, high, p_nonpositive = _gain_interval(
        y, score, (test["n_callers"] >= 2).to_numpy(), missed)
    single = []
    for name in ("cutesv_support", "caller_qual"):
        s = test[name].fillna(-1).astype(float).to_numpy()
        single.append({"name": name, "p": round(_best_precision(s, y, total_true, ref_recall), 4)})

    typed = dataset[dataset["true_sv_type"].notna()]
    pseudo = dataset[dataset["pseudo_sv_type"].notna()]
    region = bench["region_claim_by_model"]
    strata = bench["stratified"]["repeat_context"]
    cand = _read(results / "candidates" / "HG002_ONT.merge_summary.json")
    dist = _read(results / "models" / "stageA_random_forest_B+A+C+V.json")["class_distribution"]

    return {
        "chrom": TEST_CHROM,
        "threshold": float(meta.threshold),
        "ceiling": round(bench["recall_ceiling_union"], 4),
        "missed": missed,
        "truth_total": total_true,
        "ops": ops,
        "pr": curves,
        "rows": rows,
        "matched": {k: {"p": round(v["precision_at_matched_recall"], 6),
                        "gain": round(v["delta_precision_at_matched_recall"], 6)}
                    for k, v in bench["contribution"].items()},
        "comparison": {"low": round(bench["model_comparison"]["ci_low"], 4),
                       "high": round(bench["model_comparison"]["ci_high"], 4)},
        "checks": {
            "gain_low": round(low, 4), "gain_high": round(high, 4),
            "p_nonpositive": round(p_nonpositive, 3), "single": single,
            "stage_b_label_matches_caller": round(float((typed["true_sv_type"] == typed["svtype_called"]).mean()), 4),
            "stage_c_label_matches_caller": round(float((pseudo["pseudo_sv_type"] == pseudo["svtype_called"]).mean()), 4),
            "stage_b_f1": stage_b["view1_operating_points"]["rf"]["macro_f1"],
            "stage_b_n": stage_b["n_test_rows"],
            "stage_c_f1": round(stage_c["view1_operating_points"]["rf"]["macro_f1"], 3),
            "stage_c_n": stage_c["n_test_rows"],
            "low_repeat_n": strata["tr_0-0.1"]["n"], "high_repeat_n": strata["tr_0.5-1.01"]["n"],
            "region": {k: [round(v["low_precision_gain"], 3), round(v["high_precision_gain"], 3)]
                       for k, v in region.items()},
        },
        "features": [{"name": f.name, "group": f.group.value,
                      "definition": PLAIN_DEFINITIONS.get(f.name, f.definition)}
                     for f in FEATURES],
        "importance": importance,
        "samples": _samples(results, summary),
        "models": _models(results),
        "ablation": _ablation(summary),
        "splits": dist,
        "gate": summary["gate_counts"],
        "candidates": cand,
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
    rows, c = payload["rows"], payload["checks"]
    correct = sum((r["p"] >= payload["threshold"]) == (r["truth"] == 1) for r in rows)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB): {len(rows)} candidates, "
          f"{correct / len(rows):.1%} correct at {payload['threshold']:.2f}")
    print(f"gain over intersection: {payload['matched']['rf']['gain'] * 100:+.2f} pp, "
          f"95% interval [{c['gain_low'] * 100:+.1f}, {c['gain_high'] * 100:+.1f}] pp")


if __name__ == "__main__":
    main()
