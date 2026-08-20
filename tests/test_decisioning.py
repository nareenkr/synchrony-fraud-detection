from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from backend.app.explainability import ExplanationService
from backend.app.fraud.anomaly import AnomalyScorer
from backend.app.fraud.risk_engine import RiskEngine, load_risk_config
from backend.app.fraud.supervised import SupervisedModel
from backend.app.schemas import Decision
from backend.app.services.decisioning import DecisioningService
from backend.app.state import MemoryRealtimeStateStore
from training.prepare_data import _development_source, demo_scenarios, prepare_dataset
from training.replay import replay_splits


class RatioEstimator:
    """Deterministic sklearn-like test model driven by canonical risk features."""

    def predict_proba(self, frame: Any) -> np.ndarray:
        risk = np.clip(
            0.05
            + 0.35 * frame["loan_to_income_ratio"].to_numpy()
            + 0.05 * frame["failed_login_attempts_24h"].to_numpy(),
            0.0,
            1.0,
        )
        return np.column_stack([1.0 - risk, risk])


def service() -> DecisioningService:
    prepared = prepare_dataset(_development_source(80))
    matrices = replay_splits(prepared.splits)
    anomaly = AnomalyScorer().fit(matrices.train.features)
    supervised = SupervisedModel(
        estimator=RatioEstimator(),
        threshold=0.5,
        model_name="ratio-test",
        model_version="test-v1",
        manifest={},
    )
    config, version = load_risk_config("config/risk.yaml")
    return DecisioningService(
        state_store=MemoryRealtimeStateStore(),
        supervised=supervised,
        anomaly=anomaly,
        risk_engine=RiskEngine(config, version),
        explanations=ExplanationService(max_reasons=6),
    )


def test_complete_assessment_contains_all_components_and_safe_reasons() -> None:
    decisioning = service()
    event = demo_scenarios()["suspicious"][0]
    result = decisioning.assess(event, now=event.event_timestamp)

    assert 0 <= result.risk_score <= 100
    assert result.component_scores.supervised_probability > 0
    assert 0 <= result.component_scores.anomaly_score <= 1
    assert result.component_scores.behavioral_risk > 0
    assert result.reasons
    assert result.model_version == "test-v1"
    serialized = result.model_dump_json()
    assert event.user_id not in serialized
    assert event.device_id not in serialized
    assert "shap" not in serialized.lower()


def test_repeated_applications_increase_velocity_and_do_not_self_count() -> None:
    decisioning = service()
    base = demo_scenarios()["suspicious"][0]
    first_features = decisioning.score_features(base)
    first = decisioning.assess(base, now=base.event_timestamp)
    second_event = base.model_copy(
        update={
            "application_id": "APP-SUSPICIOUS-NEXT",
            "event_timestamp": base.event_timestamp + timedelta(minutes=1),
        }
    )
    second_features = decisioning.score_features(second_event)
    second = decisioning.assess(second_event, now=second_event.event_timestamp)

    assert first_features.iloc[0]["applications_last_hour"] == 0
    assert second_features.iloc[0]["applications_last_hour"] == 1
    assert second.risk_score >= first.risk_score


def test_custom_risk_thresholds_drive_decision() -> None:
    decisioning = service()
    event = demo_scenarios()["normal"][0]
    assessment = decisioning.assess(event, now=datetime(2026, 8, 20, tzinfo=UTC))
    expected = decisioning.risk_engine.decision_for(assessment.risk_score)
    assert assessment.decision is expected
    assert assessment.decision in set(Decision)
