from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.features import FEATURE_NAMES
from backend.app.fraud.anomaly import (
    AnomalyConfig,
    AnomalyScorer,
    EmpiricalQuantileCalibrator,
)
from training.prepare_data import _development_source, prepare_dataset
from training.train_anomaly import train_anomaly


def normal_training_frame(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(20260819)
    records: list[dict[str, float]] = []
    for _ in range(rows):
        amount = float(np.clip(rng.normal(5_000, 900), 1_000, 9_000))
        income = float(np.clip(rng.normal(80_000, 8_000), 45_000, 120_000))
        hour = float(rng.uniform(8, 20))
        record = dict.fromkeys(FEATURE_NAMES, 0.0)
        record.update(
            {
                "requested_loan_amount": amount,
                "income": income,
                "loan_to_income_ratio": amount / income,
                "debt_to_income_ratio": float(rng.uniform(0.05, 0.3)),
                "account_age_days": float(rng.integers(700, 2_500)),
                "bank_account_age_days": float(rng.integers(900, 3_000)),
                "origin_balance_before": float(rng.uniform(10_000, 60_000)),
                "amount_to_balance_ratio": float(rng.uniform(0.02, 0.2)),
                "balance_change_ratio": float(rng.uniform(-0.2, 0.0)),
                "applications_last_hour": float(rng.integers(0, 2)),
                "applications_last_day": float(rng.integers(0, 4)),
                "transaction_frequency_24h": float(rng.integers(1, 9)),
                "transaction_amount_deviation": float(rng.normal(0, 0.35)),
                "login_frequency_24h": float(rng.integers(1, 7)),
                "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
                "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
                "hour_of_day_deviation": float(rng.uniform(0, 2)),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=FEATURE_NAMES)


def anomaly_cases() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = normal_training_frame(1).iloc[0].to_dict()
    normal = pd.DataFrame([base], columns=FEATURE_NAMES)
    suspicious = base | {
        "requested_loan_amount": 42_000.0,
        "loan_to_income_ratio": 0.8,
        "account_age_days": 8.0,
        "applications_last_hour": 4.0,
        "transaction_frequency_24h": 28.0,
        "transaction_amount_deviation": 3.0,
        "failed_login_attempts_24h": 4.0,
        "is_new_device": 1.0,
        "unusual_login_location": 1.0,
    }
    ring = suspicious | {
        "requested_loan_amount": 95_000.0,
        "loan_to_income_ratio": 3.0,
        "account_age_days": 0.0,
        "applications_last_hour": 15.0,
        "applications_last_day": 25.0,
        "transaction_frequency_24h": 120.0,
        "transaction_amount_deviation": 12.0,
        "device_changes_30d": 10.0,
        "failed_login_attempts_24h": 15.0,
        "shared_device_user_count": 8.0,
        "shared_ip_user_count": 10.0,
        "shared_bank_user_count": 4.0,
    }
    return (
        normal,
        pd.DataFrame([suspicious], columns=FEATURE_NAMES),
        pd.DataFrame([ring], columns=FEATURE_NAMES),
    )


def test_isolation_forest_is_deterministic_normalized_and_directional() -> None:
    training = normal_training_frame()
    config = AnomalyConfig(n_estimators=120, max_samples=128, random_state=11)
    first = AnomalyScorer(config).fit(training)
    second = AnomalyScorer(config).fit(training)
    normal, suspicious, ring = anomaly_cases()

    first_scores = np.array([first.score(normal), first.score(suspicious), first.score(ring)])
    second_scores = np.array([second.score(normal), second.score(suspicious), second.score(ring)])

    np.testing.assert_allclose(first_scores, second_scores, rtol=0, atol=0)
    assert np.all((0 <= first_scores) & (first_scores <= 1))
    assert first_scores[0] < first_scores[1] <= first_scores[2]


def test_empirical_calibration_is_monotonic_and_rejects_nonfinite() -> None:
    calibrator = EmpiricalQuantileCalibrator.fit([0.4, 0.1, 0.2, 0.3])
    scores = calibrator.transform([0.05, 0.2, 0.35, 0.8])
    assert np.all(np.diff(scores) >= 0)
    assert np.all((0 <= scores) & (scores <= 1))
    with pytest.raises(ValueError, match="finite"):
        calibrator.transform([np.nan])


def test_artifact_round_trip_and_integrity_check(tmp_path: Path) -> None:
    scorer = AnomalyScorer(AnomalyConfig(n_estimators=40, max_samples=32)).fit(
        normal_training_frame(60)
    )
    artifact = tmp_path / "anomaly.joblib"
    digest = scorer.save(artifact)
    restored = AnomalyScorer.load(artifact, expected_sha256=digest)
    _, suspicious, _ = anomaly_cases()

    assert restored.score(suspicious) == scorer.score(suspicious)
    assert artifact.with_name("anomaly.joblib.sha256").is_file()
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        AnomalyScorer.load(artifact, expected_sha256=digest)


def test_score_requires_fit_exact_contract_and_single_row() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        AnomalyScorer().score(normal_training_frame(1))
    scorer = AnomalyScorer(AnomalyConfig(n_estimators=20)).fit(normal_training_frame(20))
    with pytest.raises(ValueError, match="ordered contract"):
        scorer.score(normal_training_frame(1).loc[:, reversed(FEATURE_NAMES)])
    with pytest.raises(ValueError, match="exactly one"):
        scorer.score(normal_training_frame(2))


def test_training_uses_replayed_shared_features_and_train_partition_only() -> None:
    prepared = prepare_dataset(_development_source(90))
    scorer, matrices = train_anomaly(
        prepared.splits,
        config=AnomalyConfig(n_estimators=30, max_samples=32),
    )

    assert scorer.is_fitted
    assert tuple(matrices.train.features.columns) == FEATURE_NAMES
    assert len(matrices.train.features) == len(prepared.splits.train)
    assert len(matrices.validation.features) == len(prepared.splits.validation)
    assert len(matrices.test.features) == len(prepared.splits.test)
