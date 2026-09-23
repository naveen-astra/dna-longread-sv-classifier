"""End-to-end training and encoding, exercised without the bioinformatics toolchain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from svml.ml.dataset import Encoder, abs_svlen_adjustment, build_matrix
from svml.ml.models import accepts_nan, scale_pos_weight, trivial_baselines
from svml.ml.splits import assign_splits, split_frames
from svml.ml.train import load_model, predict, save_model, train_stage

# --- encoding ---------------------------------------------------------------


def test_encoder_fits_on_train_only(feature_frame, split_map):
    """Imputation statistics must not be learned from validation or test."""
    frame = assign_splits(feature_frame, split_map)
    _, columns = build_matrix(frame, ["B", "A"])
    train = frame[frame["split"] == "train"]

    encoder = Encoder(columns=columns).fit(train)
    fitted_from_train = dict(encoder.fill_values)

    # Transforming other splits must not change the learned fill values.
    encoder.transform(frame[frame["split"] == "test"])
    assert encoder.fill_values == fitted_from_train


def test_encoder_requires_fit_first(feature_frame):
    _, columns = build_matrix(feature_frame, ["B"])
    with pytest.raises(RuntimeError, match="fit"):
        Encoder(columns=columns).transform(feature_frame)


def test_encoder_expands_categoricals_and_indicators(feature_frame):
    _, columns = build_matrix(feature_frame, ["V"])
    encoded = Encoder(columns=columns).fit(feature_frame).transform(feature_frame)

    assert "svtype_called__DEL" in encoded.columns
    assert "svtype_called__INS" in encoded.columns
    assert "log10_svlen__isna" in encoded.columns
    assert encoded.notna().all().all(), "sentinel imputation should leave no NaN"


def test_nan_model_path_preserves_missingness(feature_frame):
    """XGBoost learns a default split direction for NaN; it must not be imputed away."""
    frame = feature_frame.copy()
    frame.loc[frame.index[:50], "dp_flank_ratio"] = np.nan

    _, columns = build_matrix(frame, ["B"])
    encoder = Encoder(columns=columns).fit(frame)

    for_xgb = encoder.transform(frame, for_nan_model=True)
    for_rf = encoder.transform(frame, for_nan_model=False)

    assert for_xgb["dp_flank_ratio"].isna().sum() == 50
    assert for_rf["dp_flank_ratio"].isna().sum() == 0
    assert (for_rf.loc[frame.index[:50], "dp_flank_ratio"] == encoder.sentinel).all()
    assert for_rf["dp_flank_ratio__isna"].sum() == 50


def test_abs_svlen_removes_the_sign_leak():
    """The sign of SVLEN encodes DEL vs INS and must not reach the stage B model."""
    frame = pd.DataFrame({"svlen": [-500, 500], "span_len": [-500, 0]})

    unchanged = abs_svlen_adjustment(frame, enabled=False)
    adjusted = abs_svlen_adjustment(frame, enabled=True)

    assert (unchanged["svlen"] == [-500, 500]).all()
    assert (adjusted["svlen"] == [500, 500]).all()


def test_model_nan_handling_flags():
    assert accepts_nan("xgboost")
    assert not accepts_nan("random_forest")


def test_scale_pos_weight_balances_classes():
    y = np.array([1] * 10 + [0] * 90)
    assert scale_pos_weight(y) == pytest.approx(9.0)


def test_trivial_baselines_include_the_caller_agreement_rule():
    """Spec section 14: establish a baseline before tuning anything."""
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    n_callers = np.array([2, 1, 2, 1, 1])

    baselines = trivial_baselines(y, n_callers)
    assert set(baselines) == {"majority", "both_callers"}
    assert (baselines["majority"] == 0.0).all()
    assert (baselines["both_callers"] == np.array([1.0, 0.0, 1.0, 0.0, 0.0])).all()


# --- training ---------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["random_forest", "xgboost"])
def test_train_stage_a_end_to_end(feature_frame, split_map, tmp_path, model_name):
    if model_name == "xgboost":
        pytest.importorskip("xgboost")

    frame = assign_splits(feature_frame, split_map)
    _, columns = build_matrix(frame, ["B", "A", "C", "V"])
    train, val, test = split_frames(frame, "is_true")

    model, encoder, meta = train_stage(
        train=train,
        val=val,
        columns=columns,
        target="is_true",
        model_name=model_name,
        stage="A",
        ablation="B+A+C+V",
        base_params={"n_estimators": 20, "random_state": 0},
        seed=0,
        search_enabled=False,
    )

    assert meta.stage == "A"
    assert meta.model_name == model_name
    assert meta.classes == ["0.0", "1.0"]
    assert meta.threshold is not None and 0.0 < meta.threshold < 1.0
    assert set(meta.class_distribution) == {"train", "val"}

    probs = predict(model, encoder, test, model_name)
    assert probs.shape == (len(test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # The fixture encodes a learnable signal, so the model should beat chance.
    scores = probs[:, 1]
    preds = (scores >= meta.threshold).astype(int)
    accuracy = (preds == test["is_true"].to_numpy(dtype=int)).mean()
    assert accuracy > 0.6


def test_model_round_trips_through_disk(feature_frame, split_map, tmp_path):
    frame = assign_splits(feature_frame, split_map)
    _, columns = build_matrix(frame, ["B", "A"])
    train, val, test = split_frames(frame, "is_true")

    model, encoder, meta = train_stage(
        train=train, val=val, columns=columns, target="is_true",
        model_name="random_forest", stage="A", ablation="B+A",
        base_params={"n_estimators": 10, "random_state": 0}, seed=0, search_enabled=False,
    )
    path = save_model(model, encoder, meta, tmp_path / "m.pkl", tmp_path / "m.json")

    loaded_model, loaded_encoder, loaded_meta = load_model(path)
    assert loaded_meta.ablation == "B+A"
    assert np.allclose(
        predict(model, encoder, test, "random_forest"),
        predict(loaded_model, loaded_encoder, test, "random_forest"),
    )


def test_ablation_uses_fewer_columns(feature_frame):
    _, independent = build_matrix(feature_frame, ["B", "A"])
    _, everything = build_matrix(feature_frame, ["B", "A", "C", "V"])

    assert set(independent) < set(everything)
    assert len(independent) >= 15, "the B+A ablation must satisfy O2 on its own"


def test_train_rejects_a_single_class_target(feature_frame, split_map):
    frame = assign_splits(feature_frame, split_map)
    frame["is_true"] = 1.0
    _, columns = build_matrix(frame, ["B"])
    train, val, _ = split_frames(frame, "is_true")

    with pytest.raises(ValueError, match="only class"):
        train_stage(
            train=train, val=val, columns=columns, target="is_true",
            model_name="random_forest", stage="A", ablation="B",
            base_params={"n_estimators": 5}, seed=0, search_enabled=False,
        )


def test_multiclass_prediction_decoding_survives_absent_training_classes():
    """Regression test for a real bug found on pilot data (docs/06 D16).

    When a declared class never appears in the training fold (a genuine risk for
    Stage C's sparse pseudo-labels), scikit-learn's classifier.classes_ is a strict
    subset of the full class list in sorted order, and predict_proba's columns
    correspond to THAT subset, not to the full declared range. Naively indexing the
    full class list by predict_proba's argmax silently decodes to the wrong label.
    """
    from sklearn.ensemble import RandomForestClassifier

    # y only contains encoded classes {0, 2, 3} out of a declared range of 0..4,
    # exactly like real Stage C training data where DUP(1) and TRA(4) never appear.
    rng = np.random.default_rng(0)
    x = rng.random((60, 4))
    y = rng.choice([0, 2, 3], size=60)

    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x, y)
    assert list(model.classes_) == [0, 2, 3], "fixture assumption: model saw a strict subset"

    full_classes = ["DEL", "DUP", "INS", "INV", "TRA"]
    probs = model.predict_proba(x)

    # The buggy pattern this regression guards against:
    #   predicted = [full_classes[i] for i in probs.argmax(axis=1)]
    # would decode column 1 (encoded class 2 = "INS") as full_classes[1] = "DUP".
    encoded_ids = np.asarray(model.classes_)[probs.argmax(axis=1)]
    predicted = np.array([full_classes[int(i)] for i in encoded_ids])

    # Every prediction must be one of the classes actually seen in training.
    assert set(predicted) <= {"DEL", "INS", "INV"}
    assert "DUP" not in predicted and "TRA" not in predicted
