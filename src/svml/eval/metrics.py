"""Metrics with bootstrap confidence intervals.

Spec section 28 forbids claiming one model is superior before the comparison
demonstrates it. `compare_models` therefore refuses to name a winner when the paired
bootstrap interval spans zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from svml.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class MetricResult:
    name: str
    precision: float
    recall: float
    f1: float
    macro_f1: float
    tp: int
    fp: int
    fn: int
    per_class: list[ClassMetrics] = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    ci: dict[str, list[float]] = field(default_factory=dict)
    average_precision: float | None = None
    brier: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def fp_rate(self) -> float:
        denom = self.tp + self.fp
        return self.fp / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fp_rate"] = self.fp_rate
        return payload


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    scores: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> MetricResult:
    """Precision / recall / F1 for the positive class, plus bootstrap intervals."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )[2]

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    per_class = _per_class(y_true, y_pred, ["false", "true"])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    result = MetricResult(
        name=name,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        macro_f1=float(macro_f1),
        tp=tp,
        fp=fp,
        fn=fn,
        per_class=per_class,
        confusion=matrix,
        labels=["false", "true"],
    )

    if scores is not None and len(np.unique(y_true)) > 1:
        result.average_precision = float(average_precision_score(y_true, scores))
        try:
            result.brier = float(brier_score_loss(y_true, scores))
        except ValueError:
            result.brier = None

    if n_bootstrap:
        result.ci = _bootstrap_ci(y_true, y_pred, n_bootstrap, ci, seed, average="binary")
    return result


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    name: str,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> MetricResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    per_class = _per_class(y_true, y_pred, labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    matrix_arr = np.array(matrix)
    tp = int(np.trace(matrix_arr))
    off = int(matrix_arr.sum() - tp)

    result = MetricResult(
        name=name,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        macro_f1=float(f1),
        tp=tp,
        fp=off,
        fn=off,
        per_class=per_class,
        confusion=matrix,
        labels=labels,
    )
    if n_bootstrap:
        result.ci = _bootstrap_ci(y_true, y_pred, n_bootstrap, ci, seed, average="macro", labels=labels)
    return result


def _per_class(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[ClassMetrics]:
    label_values = list(range(len(labels))) if labels == ["false", "true"] else labels
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_values, zero_division=0
    )
    return [
        ClassMetrics(
            label=str(labels[i]),
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i in range(len(labels))
    ]


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    ci: float,
    seed: int,
    average: str,
    labels: list[str] | None = None,
) -> dict[str, list[float]]:
    """Percentile bootstrap over candidates. A difference without an interval is not a result."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {}

    samples = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            p, r, f, _ = precision_recall_fscore_support(
                y_true[idx], y_pred[idx], average=average, zero_division=0, labels=labels
            )
        except ValueError:
            continue
        samples["precision"].append(p)
        samples["recall"].append(r)
        samples["f1"].append(f)

    alpha = (1 - ci) / 2
    out: dict[str, list[float]] = {}
    for key, values in samples.items():
        if values:
            arr = np.asarray(values, dtype=float)
            out[key] = [
                float(np.quantile(arr, alpha)),
                float(np.quantile(arr, 1 - alpha)),
            ]
    return out


def pr_curve(y_true: np.ndarray, scores: np.ndarray) -> dict[str, list[float]]:
    """Full precision-recall curve. The honest headline figure (finding H2)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist() + [1.0],
    }


def precision_at_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> dict[str, float]:
    """Precision at the threshold that first reaches `target_recall`.

    This is View 2 of the evaluation protocol: comparing a filter to a baseline at
    matched recall, rather than at whichever threshold flatters the model.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    viable = np.where(recall >= target_recall)[0]
    if viable.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "threshold": float("nan")}
    idx = int(viable[np.argmax(precision[viable])])
    threshold = float(thresholds[idx]) if idx < len(thresholds) else 1.0
    return {
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "threshold": threshold,
    }


def recall_at_precision(y_true: np.ndarray, scores: np.ndarray, target_precision: float) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    viable = np.where(precision >= target_precision)[0]
    if viable.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "threshold": float("nan")}
    idx = int(viable[np.argmax(recall[viable])])
    threshold = float(thresholds[idx]) if idx < len(thresholds) else 1.0
    return {
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "threshold": threshold,
    }


def compare_models(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap on F1 difference.

    Returns `winner: None` when the interval spans zero. Spec section 28: do not claim
    one model is better before the comparison demonstrates it.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    average = "binary" if len(np.unique(y_true)) <= 2 else "macro"

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            f_a = precision_recall_fscore_support(
                y_true[idx], np.asarray(pred_a)[idx], average=average, zero_division=0
            )[2]
            f_b = precision_recall_fscore_support(
                y_true[idx], np.asarray(pred_b)[idx], average=average, zero_division=0
            )[2]
        except ValueError:
            continue
        diffs.append(f_a - f_b)

    if not diffs:
        return {"winner": None, "note": "bootstrap produced no valid resamples"}

    arr = np.asarray(diffs)
    alpha = (1 - ci) / 2
    lo, hi = float(np.quantile(arr, alpha)), float(np.quantile(arr, 1 - alpha))
    observed = float(arr.mean())

    if lo <= 0 <= hi:
        winner = None
        note = (
            f"{ci:.0%} CI for the F1 difference spans zero ([{lo:.4f}, {hi:.4f}]); "
            "no significant difference between the models."
        )
    else:
        winner = name_a if observed > 0 else name_b
        note = f"{winner} is better; {ci:.0%} CI [{lo:.4f}, {hi:.4f}] excludes zero."

    log.info("model comparison %s vs %s: %s", name_a, name_b, note)
    return {
        "model_a": name_a,
        "model_b": name_b,
        "mean_f1_difference": observed,
        "ci_low": lo,
        "ci_high": hi,
        "winner": winner,
        "note": note,
    }
