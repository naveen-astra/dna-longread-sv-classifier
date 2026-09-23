"""Report figures (spec section 18, outputs K, M, N, O).

All figures are written as PNG at a consistent size and DPI so the HTML and PDF
renderings match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from svml.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")
FIGSIZE = (9, 5.5)
DPI = 150

SV_COLOURS = {
    "DEL": "#c0392b",
    "INS": "#2980b9",
    "DUP": "#27ae60",
    "INV": "#8e44ad",
    "BND": "#f39c12",
    "TRA": "#f39c12",
}


def _save(fig, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def sv_size_histogram(frame: pd.DataFrame, out_path: str | Path) -> str:
    """SV size distribution on a log axis, split by type (output M)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sized = frame[frame["svlen"].abs() > 0]

    for svtype in sorted(sized["svtype"].dropna().unique()):
        values = sized.loc[sized["svtype"] == svtype, "svlen"].abs()
        if len(values) < 2:
            continue
        ax.hist(
            values,
            bins=np.logspace(np.log10(50), np.log10(max(values.max(), 100)), 50),
            alpha=0.55,
            label=f"{svtype} (n={len(values)})",
            color=SV_COLOURS.get(svtype),
        )

    ax.set_xscale("log")
    ax.set_xlabel("SV length (bp, log scale)")
    ax.set_ylabel("Count")
    ax.set_title("SV size distribution by type")
    ax.legend()
    return _save(fig, out_path)


def sv_type_distribution(frame: pd.DataFrame, out_path: str | Path) -> str:
    """SV type counts (output N)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts = frame["svtype"].value_counts()
    colours = [SV_COLOURS.get(t, "#7f8c8d") for t in counts.index]
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=colours)
    for i, value in enumerate(counts.to_numpy()):
        ax.text(i, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("SV type")
    ax.set_ylabel("Candidate count")
    ax.set_title("Candidate SV type distribution")
    return _save(fig, out_path)


def karyotype_plot(frame: pd.DataFrame, out_path: str | Path, contig_lengths: dict[str, int] | None = None) -> str:
    """Genome-wide candidate density by chromosome (output O)."""
    chroms = [c for c in frame["chrom"].dropna().unique()]
    chroms = sorted(chroms, key=_chrom_sort_key)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(chroms))))
    for i, chrom in enumerate(chroms):
        subset = frame[frame["chrom"] == chrom]
        length = (contig_lengths or {}).get(chrom, subset["end"].max() or 1)
        ax.hlines(i, 0, length / 1e6, color="#dfe6e9", linewidth=9, zorder=1)
        colours = [SV_COLOURS.get(t, "#7f8c8d") for t in subset["svtype"]]
        ax.scatter(
            subset["pos"] / 1e6,
            np.full(len(subset), i),
            s=7,
            c=colours,
            alpha=0.6,
            zorder=2,
        )

    ax.set_yticks(range(len(chroms)))
    ax.set_yticklabels(chroms)
    ax.set_xlabel("Position (Mb)")
    ax.set_title("Candidate SV positions across chromosomes")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colour, label=svtype)
        for svtype, colour in SV_COLOURS.items()
        if svtype in set(frame["svtype"])
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)
    return _save(fig, out_path)


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    name = str(chrom).replace("chr", "")
    if name.isdigit():
        return (0, f"{int(name):03d}")
    return (1, name)


def confusion_matrix_plot(
    matrix: list[list[int]], labels: list[str], out_path: str | Path, title: str
) -> str:
    """Confusion matrix heatmap (output K)."""
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(labels), 1.4 + 1.0 * len(labels)))
    array = np.asarray(matrix)
    sns.heatmap(
        array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, out_path)


def pr_curves_plot(
    curves: dict[str, dict[str, list[float]]],
    baseline_points: dict[str, tuple[float, float]],
    out_path: str | Path,
    recall_ceiling: float | None = None,
) -> str:
    """View 3 of the evaluation protocol: PR curves with baselines as points.

    This is the honest headline figure -- it shows the whole precision/recall tradeoff
    rather than a single threshold chosen after the fact.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for name, curve in curves.items():
        ax.plot(curve["recall"], curve["precision"], linewidth=2, label=f"{name} (curve)")

    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (name, (recall, precision)) in enumerate(baseline_points.items()):
        ax.scatter(
            recall,
            precision,
            s=110,
            marker=markers[i % len(markers)],
            edgecolor="black",
            zorder=5,
            label=f"{name} (operating point)",
        )

    if recall_ceiling is not None and np.isfinite(recall_ceiling):
        ax.axvline(
            recall_ceiling,
            linestyle="--",
            color="#636e72",
            linewidth=1.3,
            label=f"union recall ceiling ({recall_ceiling:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall: ML filters vs heuristic baselines")
    ax.legend(loc="lower left", fontsize=8)
    return _save(fig, out_path)


def feature_distributions(
    frame: pd.DataFrame, features: list[str], out_path: str | Path, target: str = "is_true"
) -> str:
    """Feature distributions split by label (report section 8, Phase 7 exit test)."""
    features = [f for f in features if f in frame.columns][:9]
    if not features:
        return ""

    rows = int(np.ceil(len(features) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(14, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    labelled = frame[frame[target].notna()] if target in frame.columns else frame

    for ax, feature in zip(axes, features, strict=False):
        series = pd.to_numeric(labelled[feature], errors="coerce")
        if target in labelled.columns and labelled[target].nunique() > 1:
            for value, colour in ((0.0, "#c0392b"), (1.0, "#27ae60")):
                subset = series[labelled[target] == value].dropna()
                if len(subset) > 1:
                    ax.hist(subset, bins=40, alpha=0.55, color=colour,
                            label="false" if value == 0.0 else "true")
            ax.legend(fontsize=8)
        else:
            ax.hist(series.dropna(), bins=40, color="#2980b9")
        ax.set_title(feature, fontsize=10)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Feature distributions by label", y=1.0)
    fig.tight_layout()
    return _save(fig, out_path)


def stratified_precision_plot(
    stratified: dict[str, Any], models: list[str], out_path: str | Path
) -> str:
    """Precision by repeat context -- the figure for the region claim (finding L1)."""
    repeat = stratified.get("repeat_context", {})
    if not repeat:
        return ""

    levels = sorted(repeat.keys())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width = 0.8 / max(1, len(models))

    for i, model in enumerate(models):
        values = [repeat[lvl].get(model, {}).get("precision", np.nan) for lvl in levels]
        ax.bar(np.arange(len(levels)) + i * width, values, width, label=model)

    ax.set_xticks(np.arange(len(levels)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels(levels, rotation=15)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Tandem-repeat overlap fraction")
    ax.set_title("Precision by repeat context (the region claim)")
    ax.legend()
    return _save(fig, out_path)


def qc_summary_plot(qc_metrics: dict[str, dict[str, Any]], out_path: str | Path) -> str:
    """Per-sample QC comparison (report section 4)."""
    samples = list(qc_metrics)
    if not samples:
        return ""

    fields = [("n_reads", "Read count"), ("read_n50", "Read N50 (bp)"),
              ("median_quality", "Median quality"), ("total_bases", "Total bases")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.ravel(), fields, strict=False):
        values = [qc_metrics[s].get(key) or 0 for s in samples]
        ax.bar(samples, values, color="#2980b9")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save(fig, out_path)
