"""Command-line interface.

Snakemake rules are thin wrappers that call these subcommands, which keeps the science
in an importable, unit-testable package rather than in the workflow (see
docs/02_repository_architecture.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from svml.utils.config import load_config
from svml.utils.logging import get_logger, setup_logging

log = get_logger("svml.cli")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--log", default=None, help="write a log file here")
    parser.add_argument("--log-level", default="INFO")


# ---------------------------------------------------------------------------
# check-env  (Phase 0 exit test)
# ---------------------------------------------------------------------------
def cmd_check_env(args: argparse.Namespace) -> int:
    from svml.utils.provenance import missing_tools, tool_versions

    cfg = load_config(args.config)
    versions = tool_versions()
    missing = missing_tools()

    print("Tool versions:")
    for name, version in versions.items():
        print(f"  {name:12s} {version or 'NOT FOUND'}")

    print(f"\nConfig digest: {cfg.digest}")
    print(f"Active regions: {cfg.active_regions()}")
    print(f"Active chromosomes: {cfg.active_chroms()}")
    print(f"Labelled samples: {cfg.labelled_samples()}")

    if missing:
        print(
            f"\nMissing tools: {', '.join(missing)}\n"
            "These have no native Windows builds. Run inside WSL2 or Docker with the "
            "svml conda environment activated (finding M4).",
            file=sys.stderr,
        )
        return 1
    print("\nAll external tools found.")
    return 0


# ---------------------------------------------------------------------------
# inspect-vcf  (Phase 6: inspect real headers before parsing, spec section 9 stage 4)
# ---------------------------------------------------------------------------
def cmd_inspect_vcf(args: argparse.Namespace) -> int:
    from svml.io.vcf import SUPPORT_FIELD_ALIASES, describe_info_fields, read_candidates

    fields = describe_info_fields(args.vcf)
    print(f"INFO fields in {args.vcf}:")
    for name, description in sorted(fields.items()):
        print(f"  {name:20s} {description[:80]}")

    found = [a for a in SUPPORT_FIELD_ALIASES if a in fields]
    print(f"\nSupporting-read aliases present: {found or 'NONE -- inspect manually'}")

    candidates = read_candidates(args.vcf, min_sv_length=args.min_sv_length)
    by_type: dict[str, int] = {}
    for cand in candidates:
        by_type[cand.svtype] = by_type.get(cand.svtype, 0) + 1
    print(f"\nCandidates >= {args.min_sv_length} bp: {len(candidates)}")
    for svtype, count in sorted(by_type.items()):
        print(f"  {svtype:5s} {count:,}")
    return 0


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def cmd_merge(args: argparse.Namespace) -> int:
    import pickle

    from svml.candidates.merge import (
        attach_provenance,
        merge_summary,
        run_survivor,
        run_truvari_collapse,
    )

    cfg = load_config(args.config)
    merge_cfg = cfg.get("merge", {})
    tool = merge_cfg.get("tool", "truvari_collapse")

    if tool == "truvari_collapse":
        truvari_cfg = cfg.get("labels.truvari", {})
        run_truvari_collapse(
            {"sniffles2": args.sniffles, "cutesv": args.cutesv},
            args.out_vcf,
            refdist=merge_cfg.get("max_distance", 500),
            pctseq=truvari_cfg.get("pctseq", 0.7),
            pctsize=truvari_cfg.get("pctsize", 0.7),
            min_sv_length=merge_cfg.get("min_sv_length", 50),
        )
    elif tool == "survivor":
        log.warning(
            "SURVIVOR emits symbolic alleles; truvari filters those whenever pctseq > 0, "
            "so labelling will find no matches unless labels.truvari.pctseq is 0."
        )
        run_survivor(
            [args.sniffles, args.cutesv],
            args.out_vcf,
            max_distance=merge_cfg.get("max_distance", 500),
            min_callers=merge_cfg.get("min_callers", 1),
            require_same_type=merge_cfg.get("require_same_type", True),
            require_same_strand=merge_cfg.get("require_same_strand", False),
            min_sv_length=merge_cfg.get("min_sv_length", 50),
        )
    else:
        raise ValueError(f"unknown merge.tool: {tool}")

    candidates = attach_provenance(
        args.out_vcf,
        {"sniffles2": args.sniffles, "cutesv": args.cutesv},
        max_distance=merge_cfg.get("max_distance", 500),
        min_sv_length=merge_cfg.get("min_sv_length", 50),
        restrict_chroms=set(cfg.active_chroms()),
    )

    # Keep only candidates inside this sample's acquired windows, away from the edges:
    # reads are sliced by overlap, so coverage tapers across a window boundary.
    from svml.utils.config import in_windows

    role = _role_for_vcf(cfg, args.sniffles)
    windows = cfg.active_windows(role)
    margin = int(cfg.get("regions.edge_margin", 0))
    before = len(candidates)
    candidates = [c for c in candidates if in_windows(c.chrom, c.pos, windows, margin)]
    log.info("window filter: kept %d of %d candidates (role=%s, margin=%d)",
             len(candidates), before, role, margin)

    with Path(args.out_pickle).open("wb") as handle:
        pickle.dump(candidates, handle)

    summary = merge_summary(candidates)
    Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def cmd_features(args: argparse.Namespace) -> int:
    import pickle

    from svml.features.extract import build_feature_matrix, load_annotations, write_matrix
    from svml.features.registry import summary as feature_summary

    cfg = load_config(args.config)
    feat_cfg = cfg.get("features", {})

    with Path(args.candidates).open("rb") as handle:
        candidates = pickle.load(handle)

    annotation_paths = {
        key: args.annotation_dir and str(Path(args.annotation_dir) / f"{key}.bed")
        for key in ("simple_repeat", "repeatmasker", "segmental_duplications", "assembly_gaps")
    }
    annotations = load_annotations(annotation_paths, reference_fasta=args.reference)

    try:
        frame = build_feature_matrix(
            candidates,
            bam_path=args.bam,
            annotations=annotations,
            sample_id=args.sample,
            w_bp=feat_cfg.get("w_bp", 200),
            w_flank=feat_cfg.get("w_flank", 1000),
            min_clip_length=feat_cfg.get("min_clip_length", 50),
            min_sv_length=cfg.get("sv_calling.min_sv_length", 50),
            max_reads_per_locus=feat_cfg.get("max_reads_per_locus", 1000),
        )
    finally:
        annotations.close()

    write_matrix(frame, args.out)
    print(json.dumps({"rows": len(frame), "features": feature_summary()}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
def cmd_label(args: argparse.Namespace) -> int:
    import pandas as pd

    from svml.io.bed import IntervalIndex
    from svml.labels.gate import apply_gate, gate_counts
    from svml.labels.match import (
        TruvariParams,
        add_pseudo_labels,
        count_missed_truth,
        labels_from_bench,
        run_bench,
    )

    cfg = load_config(args.config)
    frame = pd.read_parquet(args.features)

    confident = IntervalIndex.from_bed(args.confident_bed, name="confident_regions")
    frame = apply_gate(frame, confident)

    params = TruvariParams.from_config(cfg.get("labels.truvari", {}))
    bench_dir = run_bench(
        comparison_vcf=args.candidate_vcf,
        truth_vcf=args.truth_vcf,
        confident_bed=args.confident_bed,
        out_dir=args.bench_dir,
        params=params,
        reference_fasta=args.reference,
    )

    frame = labels_from_bench(bench_dir, frame)
    frame = add_pseudo_labels(frame)

    frame.to_parquet(args.out, index=False)
    counts = gate_counts(frame)
    counts["truth_svs_missed_by_both_callers"] = count_missed_truth(
        bench_dir,
        cfg.active_windows("labelled"),
        int(cfg.get("regions.edge_margin", 0)),
    )
    Path(args.out_summary).write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    import pandas as pd

    from svml.ml.dataset import abs_svlen_adjustment, build_matrix
    from svml.ml.splits import (
        assign_splits,
        class_distribution,
        split_frames,
        validate_splits,
        write_splits,
    )
    from svml.ml.train import save_model, train_stage

    cfg = load_config(args.config)
    stage_cfg = cfg.require(f"ml.stages.{args.stage}")
    ablation = next(
        a for a in cfg.require("ml.ablations") if a["name"] == args.ablation
    )

    frame = pd.read_parquet(args.dataset)
    frame = assign_splits(frame, cfg.split_map())
    validate_splits(frame)
    write_splits(frame, args.out_splits, cfg.split_map())

    target = stage_cfg["target"]
    frame = abs_svlen_adjustment(frame, stage_cfg.get("abs_svlen", False))

    if stage_cfg.get("classes"):
        frame = frame[frame[target].isin(stage_cfg["classes"]) | frame[target].isna()]

    distribution = class_distribution(frame, target)
    log.info("class distribution for stage %s: %s", args.stage, distribution)

    _, columns = build_matrix(frame, ablation["groups"], stage_cfg.get("exclude_features"))
    train, val, _test = split_frames(frame, target)

    model_params = cfg.get(f"ml.models.{args.model}", {})
    model, encoder, meta = train_stage(
        train=train,
        val=val,
        columns=columns,
        target=target,
        model_name=args.model,
        stage=args.stage,
        ablation=args.ablation,
        base_params=model_params,
        seed=cfg.seed,
        search_enabled=cfg.get("ml.hyperparameter_search.enabled", True),
        search_iterations=cfg.get("ml.hyperparameter_search.n_iter", 50),
        scoring=cfg.get("ml.hyperparameter_search.scoring", "f1_macro"),
        calibration_method=cfg.get("ml.calibration.method", "isotonic"),
    )
    meta.class_distribution = distribution
    save_model(model, encoder, meta, args.out_model, args.out_meta)
    print(json.dumps(meta.metadata(), indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
def cmd_benchmark(args: argparse.Namespace) -> int:
    """Evaluate a stage on the held-out test split.

    Two distinct modes, because the six-baseline comparison only makes sense for one
    of them:

      * Stage A (validity) -- the caller baselines ARE validity predictions, so the
        full union/intersection comparison applies (finding H2).
      * Stage B / C (type) -- "did Sniffles2 call this?" is not a prediction of DEL vs
        INS, so comparing type accuracy against the caller baselines would be
        meaningless. These stages report model-only multiclass metrics.
    """
    import numpy as np
    import pandas as pd

    from svml.eval.baselines import build_baselines, evaluate_all
    from svml.eval.metrics import compare_models, multiclass_metrics
    from svml.eval.strata import build_strata, evaluate_strata, region_claim_supported
    from svml.ml.dataset import abs_svlen_adjustment
    from svml.ml.splits import assign_splits
    from svml.ml.train import load_model, predict

    cfg = load_config(args.config)
    stage_cfg = cfg.require(f"ml.stages.{args.stage}")
    target = stage_cfg["target"]
    is_validity_stage = target == "is_true"

    if not stage_cfg.get("benchmarked", True):
        log.warning(
            "stage %s is not benchmarked: its labels are caller-consensus pseudo-labels, "
            "not ground truth (finding B1). Reporting descriptive metrics only.",
            args.stage,
        )

    frame = pd.read_parquet(args.dataset)
    frame = assign_splits(frame, cfg.split_map())
    frame = abs_svlen_adjustment(frame, stage_cfg.get("abs_svlen", False))

    test = frame[(frame["split"] == "test") & frame[target].notna()]
    if stage_cfg.get("classes"):
        test = test[test[target].isin(stage_cfg["classes"])]
    if test.empty:
        log.error("test split is empty for stage %s; nothing to evaluate", args.stage)
        return 1
    log.info("evaluating on %d held-out rows -- the test split is read exactly once", len(test))

    results: dict[str, Any] = {
        "stage": args.stage,
        "target": target,
        "benchmarked": bool(stage_cfg.get("benchmarked", True)),
        "config_digest": cfg.digest,
        "n_test_rows": int(len(test)),
    }
    if not results["benchmarked"]:
        results["label_provenance"] = (
            "Caller-consensus pseudo-labels; NOT GIAB ground truth. These numbers measure "
            "agreement with the callers' consensus, not accuracy (finding B1)."
        )

    bootstrap = cfg.get("evaluation.bootstrap.n_resamples", 1000)
    ci = cfg.get("evaluation.bootstrap.ci", 0.95)

    if is_validity_stage:
        y_true = test[target].to_numpy(dtype=int)
        ml_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for model_path in args.models:
            model, encoder, meta = load_model(model_path)
            probs = predict(model, encoder, test, meta.model_name)
            scores = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
            threshold = meta.threshold if meta.threshold is not None else 0.5
            key = "rf" if meta.model_name == "random_forest" else meta.model_name
            ml_predictions[key] = ((scores >= threshold).astype(int), scores)

        # Truth SVs in the test chromosomes that neither caller proposed.
        split_map = cfg.split_map()
        missed = _read_json(args.gate_summary).get("truth_svs_missed_by_both_callers", {})
        n_missed = sum(n for chrom, n in missed.items() if split_map.get(chrom) == "test")
        if not missed:
            log.warning(
                "no missed-truth count available; recall will be measured over candidates "
                "only and the union recall ceiling will read 1.0"
            )

        baselines = build_baselines(test, ml_predictions)
        results.update(
            evaluate_all(
                test, y_true, baselines,
                n_bootstrap=bootstrap, ci=ci, seed=cfg.seed,
                headline_reference=cfg.get("evaluation.headline_reference", "intersection"),
                n_missed=n_missed,
            )
        )

        strata = build_strata(test, cfg.get("evaluation.strata", {}))
        results["stratified"] = evaluate_strata(
            test, y_true, {b.name: b.y_pred for b in baselines}, strata
        )

        reference = cfg.get("evaluation.headline_reference", "intersection")
        for model_key in ml_predictions:
            results.setdefault("region_claim_by_model", {})[model_key] = region_claim_supported(
                results["stratified"], model_key, reference
            )
        if ml_predictions:
            results["region_claim"] = results["region_claim_by_model"][next(iter(ml_predictions))]

        if "rf" in ml_predictions and "xgboost" in ml_predictions:
            results["model_comparison"] = compare_models(
                y_true, ml_predictions["rf"][0], ml_predictions["xgboost"][0],
                "rf", "xgboost", n_bootstrap=bootstrap, ci=ci, seed=cfg.seed,
            )

        print(f"recall ceiling (union): {results['recall_ceiling_union']}")
        print(f"claim supported by: {results['claim_supported_by'] or 'NONE'}")

    else:
        labels = stage_cfg.get("classes") or sorted(test[target].dropna().unique().tolist())
        y_true = test[target].to_numpy()
        per_model: dict[str, Any] = {}
        predictions: dict[str, np.ndarray] = {}

        for model_path in args.models:
            model, encoder, meta = load_model(model_path)
            probs = predict(model, encoder, test, meta.model_name)
            # probs columns correspond to model.classes_ (the classes the fitted
            # model/calibrator actually saw), NOT necessarily to the full 0..N-1 range
            # meta.classes was built from. When a class is entirely absent from a
            # training fold -- a real risk for Stage C's sparse pseudo-labels at pilot
            # scale -- model.classes_ is a strict subset in sorted order, and indexing
            # meta.classes directly by argmax's positional column index silently
            # decodes to the WRONG label. Route through model.classes_ first to
            # recover the actual encoded class id, then decode that via meta.classes.
            encoded_ids = np.asarray(model.classes_)[probs.argmax(axis=1)]
            predicted = np.array([meta.classes[int(i)] for i in encoded_ids])
            key = "rf" if meta.model_name == "random_forest" else meta.model_name
            predictions[key] = predicted
            per_model[key] = multiclass_metrics(
                y_true, predicted, labels, name=key,
                n_bootstrap=bootstrap, ci=ci, seed=cfg.seed,
            ).to_dict()

        results["view1_operating_points"] = per_model
        results["labels"] = labels
        results["note"] = (
            "Caller baselines are omitted: 'did Sniffles2 call this candidate?' is not a "
            "prediction of SV type, so comparing type metrics against union/intersection "
            "would be meaningless."
        )
        if "rf" in predictions and "xgboost" in predictions:
            results["model_comparison"] = compare_models(
                y_true, predictions["rf"], predictions["xgboost"],
                "rf", "xgboost", n_bootstrap=bootstrap, ci=ci, seed=cfg.seed,
            )
        for key, metrics in per_model.items():
            print(f"{key}: macro-F1 {metrics['macro_f1']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# shap
# ---------------------------------------------------------------------------
def cmd_shap(args: argparse.Namespace) -> int:
    import pandas as pd

    from svml.explain.shap_runner import run_shap
    from svml.ml.splits import assign_splits
    from svml.ml.train import load_model

    cfg = load_config(args.config)
    model, encoder, meta = load_model(args.model)

    frame = pd.read_parquet(args.dataset)
    frame = assign_splits(frame, cfg.split_map())
    subset = frame[frame["split"] == "test"]
    if subset.empty:
        subset = frame

    from svml.ml import models as model_defs

    x = encoder.transform(subset, for_nan_model=model_defs.accepts_nan(meta.model_name))
    summary = run_shap(
        model,
        x,
        out_dir=args.out_dir,
        stage=meta.stage,
        model_name=meta.model_name,
        max_samples=cfg.get("explain.shap.max_samples", 2000),
        plots=cfg.get("explain.shap.plots", ["summary", "bar"]),
        seed=cfg.seed,
    )
    print(json.dumps({"top_features": summary["top_features"]}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> int:
    import pandas as pd

    from svml.features.registry import FEATURES
    from svml.report import plots
    from svml.report.build import build_report, html_to_pdf, write_summary_json
    from svml.utils.provenance import build_metadata

    cfg = load_config(args.config)
    metadata = build_metadata(cfg)

    by_split: dict[str, list[str]] = {}
    for chrom, split in cfg.split_map().items():
        by_split.setdefault(split, []).append(chrom)
    metadata["split_map_by_split"] = by_split

    benchmark = _read_json(args.benchmark)
    gate = _read_json(args.gate_summary)
    candidates = _read_json(args.candidate_summary)

    figures: dict[str, Any] = {"confusion": {}, "shap": []}
    fig_dir = Path(args.out_html).parent / "figures"

    if args.dataset and Path(args.dataset).exists():
        frame = pd.read_parquet(args.dataset)
        figures["sv_size"] = _rel(plots.sv_size_histogram(frame, fig_dir / "sv_size.png"), args.out_html)
        figures["sv_type"] = _rel(plots.sv_type_distribution(frame, fig_dir / "sv_type.png"), args.out_html)
        figures["karyotype"] = _rel(plots.karyotype_plot(frame, fig_dir / "karyotype.png"), args.out_html)
        top = ["dp_flank_ratio", "mean_mapq", "frac_mapq0", "support_ratio",
               "n_split_reads", "tr_overlap_frac", "segdup_overlap", "log10_svlen", "n_callers"]
        figures["features"] = _rel(
            plots.feature_distributions(frame, top, fig_dir / "features.png"), args.out_html
        )

    if benchmark:
        curves = benchmark.get("view3_pr_curves", {})
        points = {
            name: (r["recall"], r["precision"])
            for name, r in benchmark.get("view1_operating_points", {}).items()
            if name in {"sniffles2", "cutesv", "union", "intersection"}
        }
        if curves or points:
            figures["pr"] = _rel(
                plots.pr_curves_plot(
                    curves, points, fig_dir / "pr_curves.png",
                    recall_ceiling=benchmark.get("recall_ceiling_union"),
                ),
                args.out_html,
            )
        for name, result in benchmark.get("view1_operating_points", {}).items():
            if result.get("confusion"):
                figures["confusion"][name] = _rel(
                    plots.confusion_matrix_plot(
                        result["confusion"], result.get("labels", []),
                        fig_dir / f"confusion_{name}.png", f"Confusion matrix - {name}",
                    ),
                    args.out_html,
                )
        if benchmark.get("stratified"):
            figures["strata"] = _rel(
                plots.stratified_precision_plot(
                    benchmark["stratified"],
                    list(benchmark.get("contribution", {})) + ["intersection"],
                    fig_dir / "strata.png",
                ),
                args.out_html,
            )

    # Ablation table: the B+A model against the full-feature model (finding M3).
    ablation_rows = []
    for path in args.ablation_metrics or []:
        payload = _read_json(path)
        for name, result in payload.get("view1_operating_points", {}).items():
            ablation_rows.append(
                {
                    "stage": payload.get("stage"),
                    "model": name,
                    "ablation": "B+A (independent evidence only)",
                    "macro_f1": result.get("macro_f1"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                }
            )
    for name, result in benchmark.get("view1_operating_points", {}).items():
        if name in {"rf", "xgboost"}:
            ablation_rows.append(
                {
                    "stage": benchmark.get("stage"),
                    "model": name,
                    "ablation": "B+A+C+V (all features)",
                    "macro_f1": result.get("macro_f1"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                }
            )

    # Stage C: pseudo-label provenance travels with every row.
    stage_c_payload = _read_json(args.stage_c)
    stage_c_rows = []
    for name, result in stage_c_payload.get("view1_operating_points", {}).items():
        for entry in result.get("per_class", []):
            stage_c_rows.append(
                {
                    "model": name,
                    "label": entry.get("label"),
                    "precision": entry.get("precision"),
                    "recall": entry.get("recall"),
                    "f1": entry.get("f1"),
                    "support": entry.get("support"),
                }
            )

    shap_summaries = [_read_json(p) for p in (args.shap or [])]
    shap_summaries = [s for s in shap_summaries if s]
    for summary in shap_summaries:
        figures["shap"].extend(_rel(p, args.out_html) for p in summary.get("plots", []))

    # build_report expects {sample_id: {...}}; each --qc/--align-stats path is one
    # sample's own flat metrics dict (its "sample_id" field, not the argument order,
    # is the key), and multiple paths cover all samples the spec's per-sample summary
    # requires (section 19), not just the primary labelled one.
    def _by_sample_id(paths: list[str] | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in paths or []:
            data = _read_json(p)
            sample_id = data.get("sample_id") if data else None
            if sample_id:
                out[sample_id] = data
        return out

    build_report(
        out_path=args.out_html,
        metadata=metadata,
        benchmark=benchmark,
        candidate_summary=candidates,
        gate_counts=gate,
        qc_metrics=_by_sample_id(args.qc),
        align_stats=_by_sample_id(args.align_stats),
        figures=figures,
        shap_summaries=shap_summaries,
        split_info=_read_json(args.splits) or {},
        ablation={"rows": ablation_rows},
        stage_c={"rows": stage_c_rows},
        error_analysis=_read_json(args.errors) or {},
        feature_count=len(FEATURES),
    )

    write_summary_json(
        args.out_json,
        {
            "metadata": metadata,
            "candidate_summary": candidates,
            "gate_counts": gate,
            "benchmark": benchmark,
            "ablation": ablation_rows,
            "stage_c": {
                "rows": stage_c_rows,
                "label_provenance": stage_c_payload.get("label_provenance"),
            },
            "error_analysis": _read_json(args.errors),
            "shap": [{"stage": s.get("stage"), "model": s.get("model"),
                      "top_features": s.get("top_features")} for s in shap_summaries],
        },
    )

    if args.out_pdf:
        html_to_pdf(args.out_html, args.out_pdf)
    return 0


def _role_for_vcf(cfg, vcf_path: str) -> str:
    """Infer the sample role from a per-sample file name such as HG002_ONT.sniffles.vcf.gz."""
    name = Path(vcf_path).name
    for _, row in cfg.samples.iterrows():
        if name.startswith(f"{row['sample_id']}."):
            return row["role"]
    return "labelled"


def _read_json(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("could not parse %s: %s", path, exc)
        return {}


def _rel(target: str | Path, base_html: str | Path) -> str:
    if not target:
        return ""
    try:
        return str(Path(target).relative_to(Path(base_html).parent))
    except ValueError:
        return str(target)


# ---------------------------------------------------------------------------
# feature-docs  (regenerates the dictionary table from the registry)
# ---------------------------------------------------------------------------
def cmd_feature_docs(args: argparse.Namespace) -> int:
    from svml.features.registry import summary, to_markdown

    print(f"Feature counts by group: {summary()}")
    if args.out:
        Path(args.out).write_text(to_markdown(), encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(to_markdown())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="svml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-env", help="Phase 0 exit test: verify the toolchain")
    _common(p)
    p.set_defaults(func=cmd_check_env)

    p = sub.add_parser("inspect-vcf", help="inspect real caller output before parsing")
    _common(p)
    p.add_argument("vcf")
    p.add_argument("--min-sv-length", type=int, default=50)
    p.set_defaults(func=cmd_inspect_vcf)

    p = sub.add_parser("merge", help="merge caller VCFs with SURVIVOR and attach provenance")
    _common(p)
    p.add_argument("--sniffles", required=True)
    p.add_argument("--cutesv", required=True)
    p.add_argument("--out-vcf", required=True)
    p.add_argument("--out-pickle", required=True)
    p.add_argument("--out-summary", required=True)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("features", help="build the feature matrix")
    _common(p)
    p.add_argument("--candidates", required=True)
    p.add_argument("--bam", required=True)
    p.add_argument("--reference")
    p.add_argument("--annotation-dir")
    p.add_argument("--sample", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("label", help="apply the confident-region gate and truvari matching")
    _common(p)
    p.add_argument("--features", required=True)
    p.add_argument("--candidate-vcf", required=True)
    p.add_argument("--truth-vcf", required=True)
    p.add_argument("--confident-bed", required=True)
    p.add_argument("--reference")
    p.add_argument("--bench-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out-summary", required=True)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("train", help="train one (stage, model, ablation) combination")
    _common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage", required=True, choices=["A", "B", "C"])
    p.add_argument("--model", required=True, choices=["random_forest", "xgboost"])
    p.add_argument("--ablation", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--out-meta", required=True)
    p.add_argument("--out-splits", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("benchmark", help="evaluate all six baselines on the held-out test split")
    _common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--stage", default="A")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--gate-summary", help="labelling summary with missed-truth counts")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("shap", help="compute SHAP explanations")
    _common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_shap)

    p = sub.add_parser("report", help="assemble the final HTML/PDF report and summary JSON")
    _common(p)
    p.add_argument("--dataset")
    p.add_argument("--benchmark")
    p.add_argument("--gate-summary")
    p.add_argument("--candidate-summary")
    p.add_argument("--qc", nargs="*")
    p.add_argument("--align-stats", nargs="*")
    p.add_argument("--splits")
    p.add_argument("--errors")
    p.add_argument("--ablation-metrics", nargs="*")
    p.add_argument("--stage-c")
    p.add_argument("--shap", nargs="*")
    p.add_argument("--out-html", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-pdf")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("feature-docs", help="regenerate the feature dictionary table")
    _common(p)
    p.add_argument("--out")
    p.set_defaults(func=cmd_feature_docs)

    args = parser.parse_args(argv)
    setup_logging(args.log, args.log_level)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - surface a clean message to Snakemake
        log.exception("%s failed: %s", args.command, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
