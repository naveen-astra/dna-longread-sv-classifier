"""Stratified evaluation (finding L1).

The research question is about difficult and repetitive regions, so the region claim
is only testable if metrics are broken out by difficulty.

The central empirical result of the project is a single comparison:

    Is the precision gain over the intersection baseline LARGER in the high-repeat
    stratum than in the low-repeat stratum?

`region_claim_supported()` computes exactly that, and returns False when it is not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from svml.eval import metrics as M
from svml.utils.logging import get_logger

log = get_logger(__name__)


def repeat_strata(frame: pd.DataFrame, edges: list[float]) -> pd.Series:
    """Bin candidates by tandem-repeat overlap fraction."""
    labels = [f"tr_{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    return pd.cut(
        frame["tr_overlap_frac"].fillna(0.0), bins=edges, labels=labels, include_lowest=True
    )


def size_strata(frame: pd.DataFrame, edges: list[float]) -> pd.Series:
    labels = [f"{int(edges[i])}-{int(edges[i + 1])}bp" for i in range(len(edges) - 1)]
    return pd.cut(
        frame["svlen"].abs().fillna(0), bins=edges, labels=labels, include_lowest=True
    )


def build_strata(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.Series]:
    """All stratification columns defined in `evaluation.strata`."""
    out: dict[str, pd.Series] = {}

    if "tr_overlap_frac" in config and "tr_overlap_frac" in frame.columns:
        out["repeat_context"] = repeat_strata(frame, config["tr_overlap_frac"])

    if "sv_size" in config and "svlen" in frame.columns:
        out["sv_size"] = size_strata(frame, config["sv_size"])

    if config.get("segdup") == "binary" and "segdup_overlap" in frame.columns:
        out["segdup"] = np.where(frame["segdup_overlap"].fillna(0) > 0, "in_segdup", "no_segdup")

    if config.get("mapq0") == "binary" and "frac_mapq0" in frame.columns:
        out["mapq0"] = np.where(frame["frac_mapq0"].fillna(0) > 0, "has_mapq0", "no_mapq0")

    if "svtype" in frame.columns:
        out["sv_type"] = frame["svtype"]

    return {k: pd.Series(v, index=frame.index) for k, v in out.items()}


def evaluate_strata(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    strata: dict[str, pd.Series],
    min_stratum_size: int = 30,
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    """Metrics for every model within every stratum level.

    Bootstrap resamples are reduced here relative to the headline numbers, since this
    runs across many small subsets.
    """
    y_true = np.asarray(y_true, dtype=int)
    out: dict[str, Any] = {}

    for stratum_name, series in strata.items():
        levels: dict[str, Any] = {}
        for level in pd.Series(series).dropna().unique():
            mask = (series == level).to_numpy()
            n = int(mask.sum())
            if n < min_stratum_size:
                log.debug("skipping stratum %s=%s (n=%d below minimum)", stratum_name, level, n)
                continue

            level_result: dict[str, Any] = {"n": n, "n_positive": int((y_true[mask] == 1).sum())}
            for model_name, preds in predictions.items():
                result = M.binary_metrics(
                    y_true[mask],
                    np.asarray(preds)[mask],
                    name=f"{model_name}@{stratum_name}={level}",
                    n_bootstrap=n_bootstrap,
                )
                level_result[model_name] = {
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "tp": result.tp,
                    "fp": result.fp,
                    "fn": result.fn,
                    "ci": result.ci,
                }
            levels[str(level)] = level_result
        out[stratum_name] = levels

    return out


def region_claim_supported(
    stratified: dict[str, Any],
    ml_model: str,
    reference: str = "intersection",
    low_key_fragment: str = "0-0.1",
    high_key_fragment: str = "0.5-",
) -> dict[str, Any]:
    """Test the region claim: is the precision gain larger where repeats are denser?

    Returns `supported: False` when the gain is not larger in the high-repeat stratum.
    That outcome is a legitimate result and must be reported as-is (spec section 28).
    """
    repeat = stratified.get("repeat_context", {})
    if not repeat:
        return {"supported": None, "note": "no repeat_context stratification available"}

    low = _find_level(repeat, low_key_fragment)
    high = _find_level(repeat, high_key_fragment)
    if low is None or high is None:
        return {
            "supported": None,
            "note": f"could not locate low ({low_key_fragment}) and high ({high_key_fragment}) "
            f"repeat strata among {list(repeat)}",
        }

    def gain(level: dict[str, Any]) -> float | None:
        if ml_model not in level or reference not in level:
            return None
        return level[ml_model]["precision"] - level[reference]["precision"]

    low_gain, high_gain = gain(repeat[low]), gain(repeat[high])
    if low_gain is None or high_gain is None:
        return {"supported": None, "note": "model or reference missing from a stratum"}

    supported = high_gain > low_gain
    note = (
        f"precision gain over {reference}: {low_gain:+.4f} in the low-repeat stratum "
        f"({low}), {high_gain:+.4f} in the high-repeat stratum ({high}). "
        + (
            "The region claim IS supported: the gain is larger where repeats are denser."
            if supported
            else "The region claim is NOT supported: the gain is not larger in the "
            "high-repeat stratum. Report this outcome as measured."
        )
    )
    log.info("region claim for %s: %s", ml_model, note)

    return {
        "supported": bool(supported),
        "model": ml_model,
        "reference": reference,
        "low_stratum": low,
        "high_stratum": high,
        "low_precision_gain": low_gain,
        "high_precision_gain": high_gain,
        "note": note,
        "coverage_caveat": (
            "Benchmark confident regions systematically exclude many of the hardest parts "
            "of the genome, so 'high repeat' means the most repetitive regions for which "
            "truth exists, not the most repetitive regions in the genome."
        ),
    }


def _find_level(levels: dict[str, Any], fragment: str) -> str | None:
    for key in levels:
        if fragment in key:
            return key
    return None
