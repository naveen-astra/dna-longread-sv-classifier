"""Metrics, baselines and the honesty guards in the evaluation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from svml.eval.baselines import build_baselines, evaluate_all, recall_ceiling
from svml.eval.metrics import (
    binary_metrics,
    compare_models,
    precision_at_recall,
    recall_at_precision,
)
from svml.eval.strata import region_claim_supported


@pytest.fixture
def scored_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    return pd.DataFrame(
        {
            "candidate_id": [f"SV_{i}" for i in range(n)],
            "chrom": "chr20",
            "svlen": rng.integers(50, 5000, n),
            "svtype": rng.choice(["DEL", "INS"], n),
            "sniffles_support": rng.choice([np.nan, 5, 10], n),
            "cutesv_support": rng.choice([np.nan, 4, 8], n),
            "tr_overlap_frac": rng.random(n),
            "segdup_overlap": rng.choice([0.0, 0.5], n),
            "frac_mapq0": rng.choice([0.0, 0.2], n),
        }
    )


def test_binary_metrics_counts():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])

    result = binary_metrics(y_true, y_pred, "test", n_bootstrap=50)
    assert result.tp == 2
    assert result.fp == 1
    assert result.fn == 1
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)


def test_bootstrap_produces_intervals():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)

    result = binary_metrics(y_true, y_pred, "noise", n_bootstrap=200)
    for key in ("precision", "recall", "f1"):
        assert key in result.ci
        low, high = result.ci[key]
        assert low <= high


def test_recall_ceiling_caps_ml_recall(scored_frame):
    """Finding H2: the ML layer is a filter and cannot exceed the union's recall."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, len(scored_frame))

    ceiling = recall_ceiling(y_true, scored_frame)
    assert 0.0 <= ceiling <= 1.0

    scores = rng.random(len(scored_frame))
    union = (
        scored_frame["sniffles_support"].notna() | scored_frame["cutesv_support"].notna()
    ).to_numpy()
    # Any ML filter operates on the union, so its recall cannot exceed the ceiling.
    ml_pred = (union & (scores > 0.3)).astype(int)
    ml_recall = binary_metrics(y_true, ml_pred, "ml", n_bootstrap=0).recall
    assert ml_recall <= ceiling + 1e-9


def test_all_six_baselines_are_built(scored_frame):
    rng = np.random.default_rng(3)
    scores = rng.random(len(scored_frame))
    preds = (scores > 0.5).astype(int)

    baselines = build_baselines(scored_frame, {"rf": (preds, scores), "xgboost": (preds, scores)})
    names = {b.name for b in baselines}
    assert names == {"sniffles2", "cutesv", "union", "intersection", "rf", "xgboost"}


def test_intersection_is_subset_of_union(scored_frame):
    baselines = {b.name: b.y_pred for b in build_baselines(scored_frame)}
    assert (baselines["intersection"] <= baselines["union"]).all()


def test_evaluate_all_flags_unsupported_claim(scored_frame):
    """When the ML model does not beat the intersection, the claim must be marked unsupported."""
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, len(scored_frame))
    random_scores = rng.random(len(scored_frame))
    random_preds = (random_scores > 0.5).astype(int)

    results = evaluate_all(
        scored_frame,
        y_true,
        build_baselines(scored_frame, {"rf": (random_preds, random_scores)}),
        n_bootstrap=50,
    )

    assert "recall_ceiling_union" in results
    assert results["headline_reference"] == "intersection"
    assert "rf" in results["contribution"]
    # A random scorer should not support the precision claim.
    assert isinstance(results["claim_supported_by"], list)
    # The ML result must carry the recall-ceiling caveat.
    assert any("hard-capped" in n for n in results["view1_operating_points"]["rf"]["notes"])


def test_matched_recall_comparison():
    y_true = np.array([1] * 50 + [0] * 50)
    scores = np.concatenate([np.linspace(0.6, 1.0, 50), np.linspace(0.0, 0.4, 50)])

    at_recall = precision_at_recall(y_true, scores, 0.8)
    assert at_recall["recall"] >= 0.8
    assert at_recall["precision"] > 0.9

    at_precision = recall_at_precision(y_true, scores, 0.9)
    assert at_precision["precision"] >= 0.9


def test_compare_models_declines_to_pick_a_winner_on_a_tie():
    """Spec section 28: do not claim superiority before the comparison demonstrates it."""
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 2, 300)
    preds = rng.integers(0, 2, 300)

    result = compare_models(y_true, preds, preds.copy(), "rf", "xgboost", n_bootstrap=200)
    assert result["winner"] is None
    assert "spans zero" in result["note"]


def test_compare_models_identifies_a_clear_winner():
    y_true = np.array([1] * 50 + [0] * 50)
    good = y_true.copy()
    bad = 1 - y_true

    result = compare_models(y_true, good, bad, "good", "bad", n_bootstrap=200)
    assert result["winner"] == "good"


def test_region_claim_reports_false_when_gain_does_not_concentrate():
    """An unsupported region claim is a legitimate result and must be reported as such."""
    stratified = {
        "repeat_context": {
            "tr_0-0.1": {"rf": {"precision": 0.90}, "intersection": {"precision": 0.70}},
            "tr_0.5-1.01": {"rf": {"precision": 0.60}, "intersection": {"precision": 0.58}},
        }
    }
    result = region_claim_supported(stratified, "rf", "intersection")
    assert result["supported"] is False
    assert "NOT supported" in result["note"]


def test_region_claim_reports_true_when_gain_concentrates():
    stratified = {
        "repeat_context": {
            "tr_0-0.1": {"rf": {"precision": 0.92}, "intersection": {"precision": 0.90}},
            "tr_0.5-1.01": {"rf": {"precision": 0.80}, "intersection": {"precision": 0.55}},
        }
    }
    result = region_claim_supported(stratified, "rf", "intersection")
    assert result["supported"] is True
    assert "coverage_caveat" in result


def test_missed_truth_lowers_the_recall_ceiling(scored_frame):
    """Without missed-truth counting every candidate is in the union, so the 'ceiling'
    would always be 1.0. Truth SVs no caller proposed must count as false negatives."""
    frame = scored_frame.copy()
    frame["sniffles_support"] = 5.0          # every candidate called
    y_true = np.array([1] * 100 + [0] * (len(frame) - 100))

    assert recall_ceiling(y_true, frame) == pytest.approx(1.0)
    assert recall_ceiling(y_true, frame, n_missed=25) == pytest.approx(100 / 125)


def test_missed_truth_is_a_false_negative_for_every_method(scored_frame):
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, len(scored_frame))
    scores = rng.random(len(scored_frame))
    preds = (scores > 0.5).astype(int)

    base = evaluate_all(scored_frame, y_true,
                        build_baselines(scored_frame, {"rf": (preds, scores)}), n_bootstrap=0)
    missed = evaluate_all(scored_frame, y_true,
                          build_baselines(scored_frame, {"rf": (preds, scores)}),
                          n_bootstrap=0, n_missed=40)

    for name in ("sniffles2", "cutesv", "union", "intersection", "rf"):
        before = base["view1_operating_points"][name]
        after = missed["view1_operating_points"][name]
        assert after["fn"] == before["fn"] + 40
        assert after["precision"] == pytest.approx(before["precision"])   # precision unaffected
        assert after["recall"] <= before["recall"]
    assert missed["truth_svs_missed_by_both_callers"] == 40
    assert max(missed["view3_pr_curves"]["rf"]["recall"]) <= missed["recall_ceiling_union"] + 1e-9
