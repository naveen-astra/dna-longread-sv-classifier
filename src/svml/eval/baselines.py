"""The six baselines (finding H2).

The ML layer is a FILTER over the union of the two callers' candidates. It can only
remove candidates, never discover one neither caller proposed. Two consequences that
this module makes explicit rather than leaving implicit:

  * ML recall is hard-capped at the union's recall. That ceiling is printed next to
    every ML recall number.
  * The intersection ("both callers agree") is the trivial heuristic filter that the
    ML layer must beat. Beating a single caller proves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from svml.eval import metrics as M
from svml.utils.logging import get_logger

log = get_logger(__name__)

BASELINE_ORDER = ["sniffles2", "cutesv", "union", "intersection", "rf", "xgboost"]


@dataclass
class BaselinePrediction:
    name: str
    y_pred: np.ndarray
    scores: np.ndarray | None = None
    is_ml: bool = False


def caller_mask(frame: pd.DataFrame, caller: str) -> np.ndarray:
    """Boolean mask of candidates called by a given caller.

    Derived from the per-caller support columns, which are NaN when that caller did
    not report the candidate.
    """
    column = {"sniffles2": "sniffles_support", "cutesv": "cutesv_support"}[caller]
    if column not in frame.columns:
        log.warning("column %s absent; %s baseline will be empty", column, caller)
        return np.zeros(len(frame), dtype=bool)
    return frame[column].notna().to_numpy()


def build_baselines(
    frame: pd.DataFrame,
    ml_predictions: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[BaselinePrediction]:
    """Assemble all six baselines over the same candidate frame.

    `ml_predictions` maps model name -> (binary predictions, calibrated scores).
    """
    sniffles = caller_mask(frame, "sniffles2")
    cutesv = caller_mask(frame, "cutesv")

    out = [
        BaselinePrediction("sniffles2", sniffles.astype(int)),
        BaselinePrediction("cutesv", cutesv.astype(int)),
        # The union is every merged candidate by definition -- each came from at least
        # one caller -- including any whose provenance reconciliation found no per-caller
        # record (an "orphan"). Defining it via the support columns would drop those.
        BaselinePrediction("union", np.ones(len(frame), dtype=int)),
        BaselinePrediction("intersection", (sniffles & cutesv).astype(int)),
    ]

    for name, (preds, scores) in (ml_predictions or {}).items():
        out.append(BaselinePrediction(name, np.asarray(preds, dtype=int), np.asarray(scores), is_ml=True))

    return out


def recall_ceiling(y_true: np.ndarray, frame: pd.DataFrame, n_missed: int = 0) -> float:
    """Recall of the union candidate set against the truth: the hard cap on ML recall.

    `n_missed` is the number of truth SVs in the evaluated region that neither caller
    proposed (truvari's unmatched base records). Without it, recall would be measured
    only over candidates -- and since every candidate is in the union, the "ceiling"
    would always be 1.0, hiding exactly the limitation finding H2 is about.
    """
    y_true = np.asarray(y_true, dtype=int)
    found = int((y_true == 1).sum())          # every candidate is in the union
    positives = found + int(n_missed)
    if positives == 0:
        return float("nan")
    return float(found / positives)


def evaluate_all(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    baselines: list[BaselinePrediction],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    headline_reference: str = "intersection",
    n_missed: int = 0,
) -> dict[str, Any]:
    """Produce the three comparison views from docs/05_evaluation_protocol.md.

    `n_missed` truth SVs that no caller proposed are counted as false negatives for
    EVERY method, baselines and ML alike, so recall is measured against the truth rather
    than against the candidate set.
    """
    y_true = np.asarray(y_true, dtype=int)
    n_missed = int(n_missed)
    ceiling = recall_ceiling(y_true, frame, n_missed)
    n_pos_cand = int((y_true == 1).sum())
    # Candidate-level recall -> truth-level recall.
    scale = n_pos_cand / (n_pos_cand + n_missed) if (n_pos_cand + n_missed) else 1.0

    # Missed truth SVs enter every method as (true=1, predicted=0) rows.
    y_aug = np.concatenate([y_true, np.ones(n_missed, dtype=int)])

    # -- View 1: operating-point table ------------------------------------------
    results: dict[str, M.MetricResult] = {}
    for baseline in baselines:
        pred_aug = np.concatenate([np.asarray(baseline.y_pred, dtype=int),
                                   np.zeros(n_missed, dtype=int)])
        result = M.binary_metrics(
            y_aug,
            pred_aug,
            name=baseline.name,
            scores=None,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
        if baseline.scores is not None and len(np.unique(y_true)) > 1:
            cand = M.binary_metrics(y_true, baseline.y_pred, baseline.name,
                                    scores=baseline.scores, n_bootstrap=0)
            # A never-retrieved positive contributes nothing to AP; rescale accordingly.
            result.average_precision = (cand.average_precision or 0.0) * scale
            result.brier = cand.brier
            result.notes.append("Brier score is computed over candidates only.")
        if baseline.is_ml:
            result.notes.append(
                f"Recall is hard-capped at the union candidate set's recall ({ceiling:.4f}); "
                "the ML layer is a filter and cannot discover SVs neither caller proposed."
            )
        results[baseline.name] = result

    # -- View 2: matched comparison ---------------------------------------------
    reference = results.get(headline_reference)
    matched: dict[str, Any] = {}
    if reference is not None:
        for baseline in baselines:
            if not baseline.is_ml or baseline.scores is None:
                continue
            at_recall = M.precision_at_recall(
                y_true, baseline.scores, min(1.0, reference.recall / scale) if scale else 1.0
            )
            at_recall["recall"] *= scale
            at_precision = M.recall_at_precision(y_true, baseline.scores, reference.precision)
            at_precision["recall"] *= scale
            matched[baseline.name] = {
                "at_reference_recall": at_recall,
                "at_reference_precision": at_precision,
                "reference": headline_reference,
                "reference_precision": reference.precision,
                "reference_recall": reference.recall,
            }

    # -- View 3: PR curves -------------------------------------------------------
    curves = {}
    for b in baselines:
        if b.is_ml and b.scores is not None:
            curve = M.pr_curve(y_true, b.scores)
            curve["recall"] = [r * scale for r in curve["recall"]]
            curves[b.name] = curve

    # -- headline contribution ---------------------------------------------------
    contribution: dict[str, Any] = {}
    if reference is not None:
        for baseline in baselines:
            if not baseline.is_ml:
                continue
            result = results[baseline.name]
            entry = {
                "delta_f1_vs_reference": result.f1 - reference.f1,
                "delta_precision_vs_reference": result.precision - reference.precision,
                "reference": headline_reference,
            }
            if baseline.scores is not None:
                at_recall = matched[baseline.name]["at_reference_recall"]["precision"]
                entry["precision_at_matched_recall"] = at_recall
                entry["delta_precision_at_matched_recall"] = at_recall - reference.precision
                entry["claim_supported"] = bool(
                    np.isfinite(at_recall) and at_recall > reference.precision
                )
            contribution[baseline.name] = entry

    supported = [k for k, v in contribution.items() if v.get("claim_supported")]
    if contribution and not supported:
        log.warning(
            "No ML model exceeds the %s baseline's precision at matched recall. "
            "The precision claim is NOT supported by this run and the report must say so.",
            headline_reference,
        )

    return {
        "recall_ceiling_union": ceiling,
        "truth_svs_missed_by_both_callers": n_missed,
        "truth_positives_total": n_pos_cand + n_missed,
        "headline_reference": headline_reference,
        "view1_operating_points": {k: v.to_dict() for k, v in results.items()},
        "view2_matched": matched,
        "view3_pr_curves": curves,
        "contribution": contribution,
        "claim_supported_by": supported,
    }
