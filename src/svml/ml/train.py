"""Training: fit, search, calibrate, serialise.

Discipline enforced here (docs/05_evaluation_protocol.md, "Execution discipline"):
hyperparameter search and threshold selection use the VALIDATION split only. The test
split is never read by this module.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler

from svml.ml import models
from svml.ml.dataset import Encoder
from svml.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainedModel:
    model_name: str
    stage: str
    ablation: str
    classes: list[str]
    columns: list[str]
    best_params: dict[str, Any]
    val_score: float
    threshold: float | None
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    seed: int = 42

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def _fit_encoder(train: pd.DataFrame, columns: list[str]) -> Encoder:
    return Encoder(columns=list(columns)).fit(train)


def _prepare(
    encoder: Encoder, frame: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    return encoder.transform(frame, for_nan_model=models.accepts_nan(model_name))


def train_stage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    columns: list[str],
    target: str,
    model_name: str,
    stage: str,
    ablation: str,
    base_params: dict[str, Any],
    seed: int = 42,
    search_enabled: bool = True,
    search_iterations: int = 50,
    scoring: str = "f1_macro",
    calibration_method: str = "isotonic",
) -> tuple[Any, Encoder, TrainedModel]:
    """Fit one (stage, model, ablation) combination and calibrate it on validation."""
    if train.empty:
        raise ValueError(f"no training rows for stage {stage}; check the confident-region gate")
    if val.empty:
        raise ValueError(f"no validation rows for stage {stage}; check regions.tier2 in config")

    y_train_raw = train[target]
    y_val_raw = val[target]

    classes = sorted(pd.concat([y_train_raw, y_val_raw]).dropna().unique().tolist(), key=str)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = y_train_raw.map(class_to_idx).to_numpy(dtype=int)
    y_val = y_val_raw.map(class_to_idx).to_numpy(dtype=int)

    log.info(
        "stage=%s model=%s ablation=%s | train=%d val=%d classes=%s",
        stage, model_name, ablation, len(train), len(val), classes,
    )
    if len(classes) < 2:
        raise ValueError(
            f"stage {stage} has only class {classes} present; cannot train a classifier. "
            "Check the label distribution before proceeding."
        )

    encoder = _fit_encoder(train, columns)
    x_train = _prepare(encoder, train, model_name)
    x_val = _prepare(encoder, val, model_name)

    params = dict(base_params)
    if model_name == "xgboost" and len(classes) == 2:
        params.setdefault("scale_pos_weight", models.scale_pos_weight(y_train.astype(float)))

    best_params = params
    best_score = -np.inf

    if search_enabled:
        space = models.search_space(model_name)
        sampler = ParameterSampler(space, n_iter=search_iterations, random_state=seed)
        for i, candidate in enumerate(sampler):
            trial = {**params, **candidate}
            model = models.build_model(model_name, trial, seed, len(classes))
            model.fit(x_train, y_train)
            score = _score(model, x_val, y_val, scoring)
            if score > best_score:
                best_score, best_params = score, trial
            if i and i % 10 == 0:
                log.info("  search %d/%d, best %s=%.4f", i, search_iterations, scoring, best_score)
        log.info("search complete: best %s=%.4f", scoring, best_score)

    final = models.build_model(model_name, best_params, seed, len(classes))
    final.fit(x_train, y_train)
    if not search_enabled:
        best_score = _score(final, x_val, y_val, scoring)

    calibrated = _calibrate(final, x_val, y_val, calibration_method)

    threshold = None
    if len(classes) == 2:
        threshold = select_threshold(calibrated, x_val, y_val)
        log.info("selected operating threshold %.3f on the validation split", threshold)

    meta = TrainedModel(
        model_name=model_name,
        stage=stage,
        ablation=ablation,
        classes=[str(c) for c in classes],
        columns=list(encoder.columns),
        best_params={k: _jsonable(v) for k, v in best_params.items()},
        val_score=float(best_score),
        threshold=threshold,
        class_distribution={
            "train": _counts(y_train_raw),
            "val": _counts(y_val_raw),
        },
        seed=seed,
    )
    return calibrated, encoder, meta


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=True).items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _score(model, x_val, y_val, scoring: str) -> float:
    preds = model.predict(x_val)
    average = "macro" if scoring == "f1_macro" else "binary"
    return float(f1_score(y_val, preds, average=average, zero_division=0))


def _calibrate(model, x_val, y_val, method: str):
    """Calibrate probabilities on the validation split.

    Spec section 10 requires a confidence output, and an uncalibrated probability is
    not a confidence. `cv="prefit"` means the base model keeps its training fit and
    only the calibration map is learned here.
    """
    try:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrated.fit(x_val, y_val)
        return calibrated
    except (ValueError, TypeError) as exc:
        # Isotonic needs enough samples per class; fall back rather than fail the run.
        log.warning("calibration with %s failed (%s); using the uncalibrated model", method, exc)
        return model


def select_threshold(model, x_val, y_val) -> float:
    """Pick the binary threshold maximising macro-F1 on validation.

    Chosen on validation precisely so the benchmark comparison does not depend on a
    threshold tuned against test data. The benchmark reports full PR curves anyway
    (finding H2), so this threshold only sets the single reported operating point.
    """
    probs = model.predict_proba(x_val)[:, 1]
    best_threshold, best_f1 = 0.5, -np.inf
    for threshold in np.linspace(0.05, 0.95, 91):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_val, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return best_threshold


def predict(model, encoder: Encoder, frame: pd.DataFrame, model_name: str) -> np.ndarray:
    x = _prepare(encoder, frame, model_name)
    return model.predict_proba(x)


def save_model(
    model,
    encoder: Encoder,
    meta: TrainedModel,
    model_path: str | Path,
    meta_path: str | Path | None = None,
) -> Path:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "encoder": encoder, "meta": meta}, handle)

    meta_file = Path(meta_path) if meta_path else path.with_suffix(".json")
    meta_file.write_text(json.dumps(meta.metadata(), indent=2, default=str), encoding="utf-8")
    log.info("saved model to %s", path)
    return path


def load_model(model_path: str | Path) -> tuple[Any, Encoder, TrainedModel]:
    with Path(model_path).open("rb") as handle:
        payload = pickle.load(handle)
    return payload["model"], payload["encoder"], payload["meta"]
