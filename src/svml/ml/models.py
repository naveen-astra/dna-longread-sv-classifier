"""Model definitions and hyperparameter search spaces.

Random Forest and XGBoost only (spec section 10). Neither is assumed superior before
the comparison is run (spec section 28); `svml.eval.metrics` decides that with a paired
bootstrap and declines to declare a winner when the intervals overlap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from svml.utils.logging import get_logger

log = get_logger(__name__)

MODEL_NAMES = ("random_forest", "xgboost")


def accepts_nan(model_name: str) -> bool:
    """XGBoost learns a default split direction for NaN; scikit-learn's forest cannot."""
    return model_name == "xgboost"


def build_model(model_name: str, params: dict[str, Any], seed: int, n_classes: int):
    if model_name == "random_forest":
        return _random_forest(params, seed)
    if model_name == "xgboost":
        return _xgboost(params, seed, n_classes)
    raise ValueError(f"unknown model: {model_name}. Expected one of {MODEL_NAMES}")


def _random_forest(params: dict[str, Any], seed: int) -> RandomForestClassifier:
    kwargs: dict[str, Any] = {
        "n_estimators": params.get("n_estimators", 500),
        "max_depth": params.get("max_depth"),
        "min_samples_leaf": params.get("min_samples_leaf", 1),
        "max_features": params.get("max_features", "sqrt"),
        "class_weight": params.get("class_weight", "balanced"),
        "random_state": params.get("random_state", seed),
        "n_jobs": params.get("n_jobs", -1),
    }
    return RandomForestClassifier(**kwargs)


def _xgboost(params: dict[str, Any], seed: int, n_classes: int):
    from xgboost import XGBClassifier

    kwargs: dict[str, Any] = {
        "n_estimators": params.get("n_estimators", 500),
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.1),
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
        "min_child_weight": params.get("min_child_weight", 1),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "random_state": params.get("random_state", seed),
        "n_jobs": params.get("n_jobs", -1),
        "tree_method": "hist",
        "eval_metric": "logloss" if n_classes <= 2 else "mlogloss",
    }
    if n_classes > 2:
        kwargs["objective"] = "multi:softprob"
        kwargs["num_class"] = n_classes
    else:
        kwargs["objective"] = "binary:logistic"
        if "scale_pos_weight" in params:
            kwargs["scale_pos_weight"] = params["scale_pos_weight"]
    return XGBClassifier(**kwargs)


def search_space(model_name: str) -> dict[str, list[Any]]:
    """Randomised-search space. Searched on the validation split only."""
    if model_name == "random_forest":
        return {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [None, 8, 12, 20, 30],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3],
        }
    if model_name == "xgboost":
        return {
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [0.5, 1.0, 5.0],
        }
    raise ValueError(f"unknown model: {model_name}")


def scale_pos_weight(y: np.ndarray) -> float:
    """Ratio of negatives to positives, for XGBoost's binary imbalance handling."""
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0:
        log.warning("no positive examples present; scale_pos_weight defaults to 1.0")
        return 1.0
    return max(negatives / positives, 1e-6)


def trivial_baselines(y_true: np.ndarray, n_callers: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Predictions a model must beat before any tuning is worthwhile (spec section 14).

    `majority` is the degenerate baseline. `both_callers` is the intersection rule --
    the real heuristic competitor (finding H2).
    """
    values, counts = np.unique(y_true[~np.isnan(y_true)], return_counts=True)
    majority_class = values[int(np.argmax(counts))] if values.size else 0.0

    out = {"majority": np.full(y_true.shape, majority_class, dtype=float)}
    if n_callers is not None:
        out["both_callers"] = (np.asarray(n_callers) >= 2).astype(float)
    return out
