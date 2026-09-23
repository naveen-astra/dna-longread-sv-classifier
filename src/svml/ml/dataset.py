"""Dataset assembly: encoding, missing-value handling and matrix construction.

Imputation statistics are fit on the TRAINING split only and applied unchanged to
validation and test, so no test-set information leaks into preprocessing
(docs/04_ml_task_definition.md, leakage control 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from svml.features.registry import (
    CHROM_CLASSES,
    FEATURES_BY_NAME,
    SVTYPE_CLASSES,
    Missing,
    feature_names,
    indicator_name,
)
from svml.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Encoder:
    """Fit-on-train encoder for one ablation group set.

    Holds the one-hot layout and the per-column fill values so the exact same
    transformation can be replayed on validation, test and inference data.
    """

    columns: list[str]
    fill_values: dict[str, float] = field(default_factory=dict)
    sentinel: float = -1.0
    fitted: bool = False

    def fit(self, frame: pd.DataFrame) -> Encoder:
        encoded = _encode(frame, self.columns)
        for col in encoded.columns:
            if col.endswith("__isna"):
                continue
            series = pd.to_numeric(encoded[col], errors="coerce")
            median = series.median()
            self.fill_values[col] = float(median) if pd.notna(median) else 0.0
        self.fitted = True
        log.info("encoder fitted on %d training rows, %d columns", len(frame), len(encoded.columns))
        return self

    def transform(self, frame: pd.DataFrame, for_nan_model: bool = False) -> pd.DataFrame:
        """Encode and impute.

        `for_nan_model=True` leaves NaN in place for XGBoost, which learns a default
        split direction per feature and handles missingness better than imputation.
        Random Forest cannot accept NaN, so it gets the sentinel plus the indicator.
        """
        if not self.fitted:
            raise RuntimeError("Encoder.fit() must be called on the training split first")

        encoded = _encode(frame, self.columns)
        if for_nan_model:
            return encoded.astype(float)

        for col in encoded.columns:
            if col.endswith("__isna"):
                continue
            feat_name = col.split("__")[0]
            feat = FEATURES_BY_NAME.get(feat_name)
            # Distance features use a large sentinel; everything else uses a negative
            # sentinel that sits outside every feature's valid range, so the forest can
            # split missing away from real values.
            if feat is not None and feat.missing is Missing.SENTINEL:
                continue
            encoded[col] = encoded[col].fillna(self.sentinel)
        return encoded.astype(float)


def _encode(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot categoricals, add missingness indicators, return numeric columns only."""
    out = pd.DataFrame(index=frame.index)

    for name in columns:
        feat = FEATURES_BY_NAME[name]

        if feat.categorical:
            classes = CHROM_CLASSES if name == "chrom_class" else SVTYPE_CLASSES
            values = frame[name].astype("object") if name in frame.columns else pd.Series("", index=frame.index)
            for cls in classes:
                out[f"{name}__{cls}"] = (values == cls).astype(float)
            continue

        series = pd.to_numeric(frame[name], errors="coerce") if name in frame.columns else pd.Series(np.nan, index=frame.index)
        out[name] = series
        if feat.needs_indicator:
            out[indicator_name(name)] = series.isna().astype(float)

    return out


def build_matrix(
    frame: pd.DataFrame,
    groups: list[str],
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select the feature columns for an ablation and return (frame, column names)."""
    exclude_set = set(exclude or [])
    names = [n for n in feature_names(groups) if n not in exclude_set]
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise KeyError(f"feature matrix is missing expected columns: {missing}")
    return frame, names


def prepare_target(
    frame: pd.DataFrame,
    target: str,
    classes: list[str] | None = None,
) -> tuple[pd.Series, list[str] | None]:
    """Extract the target column, restricting to `classes` where the stage defines them."""
    series = frame[target]
    if classes is None:
        return series.astype(float), None

    mask = series.isin(classes)
    dropped = int((~mask & series.notna()).sum())
    if dropped:
        log.info("dropped %d rows whose %s is outside %s", dropped, target, classes)
    return series.where(mask), classes


def abs_svlen_adjustment(frame: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    """Strip the sign of SVLEN so it cannot leak the DEL/INS target (finding H1).

    `log10_svlen` is already built from |SVLEN|, but `svlen` and `span_len` can still
    carry the sign. Stage B sets `abs_svlen: true` in config to neutralise it.
    """
    if not enabled:
        return frame
    frame = frame.copy()
    for col in ("svlen", "span_len"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").abs()
    return frame


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    log.info("loaded dataset %s: %d rows x %d columns", path, len(frame), frame.shape[1])
    return frame


def save_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    log.info("wrote dataset to %s", out)
    return out
