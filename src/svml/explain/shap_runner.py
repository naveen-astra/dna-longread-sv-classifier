"""SHAP explainability (spec section 17).

SHAP is an explanation layer. It does not alter predictions and does not replace the
classifier.

Phase 11 exit test: SHAP's top features must be consistent with the biological meaning
recorded in the feature registry. A disagreement is either a wrong rationale in the
dictionary or a bug in the feature, and both need resolving before the report --
`consistency_check()` surfaces them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from svml.features.registry import FEATURES_BY_NAME  # noqa: E402
from svml.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def _unwrap(model):
    """Recover the underlying tree model from a CalibratedClassifierCV wrapper."""
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0]
        for attr in ("estimator", "base_estimator"):
            if hasattr(inner, attr):
                return getattr(inner, attr)
    return model


def run_shap(
    model,
    x: pd.DataFrame,
    out_dir: str | Path,
    stage: str,
    model_name: str,
    max_samples: int = 2000,
    plots: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute SHAP values and write the requested plots."""
    import shap

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = plots or ["summary", "bar", "beeswarm"]

    if len(x) > max_samples:
        x = x.sample(max_samples, random_state=seed)
        log.info("subsampled to %d rows for SHAP", max_samples)

    base = _unwrap(model)
    try:
        explainer = shap.TreeExplainer(base)
        values = explainer.shap_values(x)
    except Exception as exc:  # noqa: BLE001 - explainer support varies by model version
        log.warning("TreeExplainer failed (%s); falling back to the generic Explainer", exc)
        explainer = shap.Explainer(base, x)
        values = explainer(x).values

    # Multiclass returns one matrix per class; use the positive/last class for ranking.
    matrix = values[-1] if isinstance(values, list) else values
    if isinstance(matrix, np.ndarray) and matrix.ndim == 3:
        matrix = matrix[:, :, -1]

    importance = (
        pd.Series(np.abs(matrix).mean(axis=0), index=x.columns)
        .sort_values(ascending=False)
    )

    prefix = f"{stage}_{model_name}"
    written: list[str] = []

    for kind in plots:
        path = out_dir / f"shap_{kind}_{prefix}.png"
        try:
            plt.figure(figsize=(9, 7))
            if kind == "bar":
                shap.summary_plot(matrix, x, plot_type="bar", show=False, max_display=20)
            elif kind in {"summary", "beeswarm"}:
                shap.summary_plot(matrix, x, show=False, max_display=20)
            elif kind == "dependence":
                plt.close()
                for feature in importance.index[:3]:
                    dep_path = out_dir / f"shap_dependence_{feature}_{prefix}.png"
                    plt.figure(figsize=(7, 5))
                    shap.dependence_plot(feature, matrix, x, show=False)
                    plt.tight_layout()
                    plt.savefig(dep_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    written.append(str(dep_path))
                continue
            else:
                plt.close()
                continue
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            written.append(str(path))
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP plot '%s' failed: %s", kind, exc)
            plt.close()

    summary = {
        "stage": stage,
        "model": model_name,
        "n_samples": int(len(x)),
        "global_importance": {k: float(v) for k, v in importance.head(30).items()},
        "top_features": importance.head(10).index.tolist(),
        "plots": written,
        "consistency": consistency_check(importance),
    }

    (out_dir / f"shap_summary_{prefix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("SHAP top features for %s: %s", prefix, summary["top_features"][:5])
    return summary


def consistency_check(importance: pd.Series, top_n: int = 10) -> dict[str, Any]:
    """Pair each top feature with its recorded biological meaning for manual review.

    Cannot be fully automated -- the check is whether the learned relationship matches
    the stated biology, which a human has to read. This surfaces the pairs to read.
    """
    rows = []
    for name in importance.head(top_n).index:
        base_name = str(name).split("__")[0]
        feature = FEATURES_BY_NAME.get(base_name)
        rows.append(
            {
                "feature": str(name),
                "group": feature.group.value if feature else "derived",
                "mean_abs_shap": float(importance[name]),
                "stated_meaning": feature.meaning if feature else "(indicator or one-hot column)",
            }
        )

    caller_driven = sum(1 for r in rows if r["group"] in {"C", "V"})
    note = (
        f"{caller_driven} of the top {len(rows)} features come from the caller-reported "
        "groups (C/V). If the model depends mainly on these, it is closer to a "
        "recalibration of the callers than to independent evidence -- compare against "
        "the B+A ablation before claiming otherwise (finding M3)."
    )
    return {"top_features": rows, "caller_driven_count": caller_driven, "note": note}


def explain_prediction(
    model,
    x_row: pd.DataFrame,
    out_path: str | Path | None = None,
) -> dict[str, float]:
    """Per-prediction explanation, for the 'why did it predict this' trace."""
    import shap

    base = _unwrap(model)
    explainer = shap.TreeExplainer(base)
    values = explainer.shap_values(x_row)
    matrix = values[-1] if isinstance(values, list) else values
    if isinstance(matrix, np.ndarray) and matrix.ndim == 3:
        matrix = matrix[:, :, -1]

    contributions = dict(zip(x_row.columns, np.asarray(matrix)[0], strict=False))
    ranked = dict(sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True))

    if out_path is not None:
        Path(out_path).write_text(
            json.dumps({k: float(v) for k, v in ranked.items()}, indent=2), encoding="utf-8"
        )
    return {k: float(v) for k, v in ranked.items()}
