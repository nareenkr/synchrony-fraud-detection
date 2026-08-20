"""Integration tests for deterministic training and versioned classifier bundles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.features.contract import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from backend.app.fraud.supervised import BundleValidationError, load_classifier_bundle
from training.replay import ModelMatrix, ModelMatrixSplits
from training.train_classifier import persist_classifier_bundle, train_supervised_models


def _matrix(rows: int, offset: int) -> ModelMatrix:
    rng = np.random.default_rng(1234 + offset)
    labels = np.tile([0, 0, 0, 1], rows // 4 + 1)[:rows]
    risk = np.clip(labels * 0.7 + rng.normal(0.0, 0.08, rows), 0.0, 1.0)
    values = np.zeros((rows, len(FEATURE_NAMES)), dtype=np.float64)
    values[:, FEATURE_NAMES.index("requested_loan_amount")] = 10_000 + risk * 30_000
    values[:, FEATURE_NAMES.index("income")] = 80_000 - risk * 25_000
    values[:, FEATURE_NAMES.index("loan_to_income_ratio")] = 0.1 + risk
    values[:, FEATURE_NAMES.index("account_age_days")] = 2_000 - risk * 1_500
    values[:, FEATURE_NAMES.index("applications_last_hour")] = np.clip(risk * 8, 0, None)
    values[:, FEATURE_NAMES.index("shared_device_user_count")] = np.clip(risk * 5, 0, None)
    frame = pd.DataFrame(values, columns=FEATURE_NAMES)
    return ModelMatrix(
        features=frame,
        labels=pd.Series(labels, dtype=int),
        application_ids=pd.Series([f"app-{offset}-{index}" for index in range(rows)]),
    )


def _matrices() -> ModelMatrixSplits:
    return ModelMatrixSplits(train=_matrix(48, 0), validation=_matrix(20, 1), test=_matrix(20, 2))


def test_training_fits_baseline_and_challenger_deterministically() -> None:
    first = train_supervised_models(_matrices(), seed=42, max_false_positive_rate=0.10)
    second = train_supervised_models(_matrices(), seed=42, max_false_positive_rate=0.10)

    assert {candidate.name for candidate in first.candidates} == {
        "logistic_regression",
        "xgboost",
    }
    assert first.selected.name == second.selected.name
    assert first.selected.threshold.threshold == pytest.approx(second.selected.threshold.threshold)
    assert first.test_metrics == second.test_metrics
    assert first.test_metrics["row_count"] == 20
    assert first.test_metrics["roc_auc"] is not None


def test_single_class_validation_and_test_are_retained_as_undefined_metrics() -> None:
    matrices = _matrices()
    zeros = pd.Series(np.zeros(20, dtype=int))
    single_class = ModelMatrixSplits(
        train=matrices.train,
        validation=ModelMatrix(
            matrices.validation.features, zeros, matrices.validation.application_ids
        ),
        test=ModelMatrix(matrices.test.features, zeros, matrices.test.application_ids),
    )

    result = train_supervised_models(single_class, seed=42)

    assert result.selected.threshold.status == "validation_has_no_positive_labels"
    assert result.test_metrics["roc_auc"] is None
    assert result.test_metrics["pr_auc"] is None
    assert result.test_metrics["recall"] is None


def test_single_class_training_fails_with_actionable_error() -> None:
    matrices = _matrices()
    invalid_train = ModelMatrix(
        matrices.train.features,
        pd.Series(np.zeros(len(matrices.train.labels), dtype=int)),
        matrices.train.application_ids,
    )

    with pytest.raises(ValueError, match="training partition must contain both"):
        train_supervised_models(
            ModelMatrixSplits(invalid_train, matrices.validation, matrices.test), seed=42
        )


def test_persisted_bundle_validates_hash_schema_and_inference(tmp_path: Path) -> None:
    matrices = _matrices()
    result = train_supervised_models(matrices, seed=42)
    bundle_dir = tmp_path / "classifier-test-v1"
    manifest = persist_classifier_bundle(
        result,
        bundle_dir,
        model_version="classifier-test-v1",
        seed=42,
        dataset_manifest={"schema_version": 1, "content_sha256": "abc"},
        prepared_manifest_sha256=None,
        overlap_audit={},
        max_false_positive_rate=0.10,
        max_review_rate=None,
    )

    loaded = load_classifier_bundle(bundle_dir)

    assert loaded.model_version == "classifier-test-v1"
    assert loaded.manifest["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert set(manifest["artifact_hashes"]) == {"classifier.joblib", "feature_schema.json"}
    np.testing.assert_allclose(
        loaded.predict_proba(matrices.test.features),
        result.selected.estimator.predict_proba(matrices.test.features)[:, 1],
    )

    schema_path = bundle_dir / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["schema_version"] = "wrong"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(BundleValidationError, match="hash mismatch"):
        load_classifier_bundle(bundle_dir)
