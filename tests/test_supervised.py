"""Tests for supervised metrics, operating thresholds, and inference validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from backend.app.features.contract import FEATURE_NAMES
from backend.app.fraud.supervised import (
    SupervisedModel,
    binary_classification_metrics,
    tune_threshold,
)


def _feature_frame(rows: int = 8) -> pd.DataFrame:
    values = np.zeros((rows, len(FEATURE_NAMES)), dtype=np.float64)
    values[:, FEATURE_NAMES.index("requested_loan_amount")] = np.arange(rows) * 1_000
    values[:, FEATURE_NAMES.index("income")] = 50_000
    values[:, FEATURE_NAMES.index("loan_to_income_ratio")] = np.arange(rows) / 10
    return pd.DataFrame(values, columns=FEATURE_NAMES)


def test_threshold_tuning_prioritizes_recall_within_false_positive_constraint() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.8, 0.2, 0.9, 0.7]

    selected = tune_threshold(labels, probabilities, max_false_positive_rate=0.0)

    assert selected.feasible is True
    assert selected.threshold == pytest.approx(0.9)
    assert selected.metrics["recall"] == pytest.approx(0.5)
    assert selected.metrics["false_positive_rate"] == 0.0
    assert any(row["threshold"] == 0.8 and not row["feasible"] for row in selected.table)


def test_review_capacity_is_a_real_threshold_constraint() -> None:
    selected = tune_threshold(
        [0, 0, 1, 1],
        [0.8, 0.2, 0.9, 0.7],
        max_false_positive_rate=None,
        max_review_rate=0.25,
    )

    assert selected.feasible is True
    assert selected.metrics["review_rate"] <= 0.25
    assert selected.metrics["recall"] == pytest.approx(0.5)


def test_threshold_uses_allowed_fpr_budget_for_temporal_margin() -> None:
    normal_scores = np.linspace(0.01, 0.18, 18)
    selected = tune_threshold(
        [0] * 18 + [1, 1],
        [*normal_scores, 0.90, 0.95],
        max_false_positive_rate=0.10,
    )

    assert selected.metrics["recall"] == 1.0
    assert selected.metrics["false_positive_rate"] <= 0.10
    assert selected.threshold == pytest.approx(0.18)


def test_single_class_partition_marks_unavailable_metrics_instead_of_fabricating() -> None:
    metrics = binary_classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["recall"] is None
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["precision_recall_curve"] is None
    assert set(metrics["undefined_metrics"]) >= {"roc_auc", "pr_auc", "recall"}

    selection = tune_threshold([0, 0, 0], [0.1, 0.2, 0.3])
    assert selection.status == "validation_has_no_positive_labels"
    assert selection.metrics["review_rate"] == 0.0


def test_runtime_wrapper_enforces_exact_shared_feature_contract() -> None:
    frame = _feature_frame()
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    estimator = LogisticRegression(random_state=7).fit(frame, labels)
    model = SupervisedModel(estimator, 0.5, "test", "test-v1", {})

    probabilities = model.predict_proba(frame)

    assert probabilities.shape == (len(frame),)
    assert np.isfinite(probabilities).all()
    with pytest.raises(ValueError, match="ordered contract"):
        model.predict_proba(frame.loc[:, reversed(FEATURE_NAMES)])


def test_f1_is_zero_not_missing_when_precision_and_recall_are_both_zero() -> None:
    metrics = binary_classification_metrics([0, 1], [0.9, 0.1], 0.5)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
