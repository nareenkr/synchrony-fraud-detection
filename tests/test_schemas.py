from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.schemas import (
    ComponentScores,
    Decision,
    FraudAssessment,
    LoanApplicationEvent,
)


def valid_event(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "application_id": "APP-001",
        "user_id": "USER-001",
        "event_timestamp": datetime(2026, 8, 19, 10, tzinfo=UTC),
        "requested_loan_amount": 5_000.0,
    }
    payload.update(overrides)
    return payload


def component_scores() -> ComponentScores:
    return ComponentScores(
        supervised_probability=0.2,
        anomaly_score=0.1,
        behavioral_risk=0.3,
        graph_risk=0.0,
    )


def test_new_user_with_missing_optional_observations_is_valid() -> None:
    event = LoanApplicationEvent.model_validate(valid_event())
    assert event.device_id is None
    assert event.income is None
    assert event.failed_login_attempts_24h is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_timestamp", datetime(2026, 8, 19, 10)),
        ("requested_loan_amount", 0.0),
        ("requested_loan_amount", float("inf")),
        ("ip_address", "999.1.2.3"),
        ("failed_login_attempts_24h", -1),
        ("application_id", "contains spaces"),
    ],
)
def test_malformed_or_unsafe_event_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        LoanApplicationEvent.model_validate(valid_event(**{field: value}))


def test_unknown_request_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        LoanApplicationEvent.model_validate(valid_event(secret_note="do not accept"))


def test_assessment_transport_does_not_hardcode_configurable_thresholds() -> None:
    # A deployment may configure 25/60 instead of 40/70. The RiskEngine owns
    # score/decision consistency; the transport schema must accept its output.
    result = FraudAssessment(
        application_id="APP-001",
        assessed_at=datetime(2026, 8, 19, 10, tzinfo=UTC),
        risk_score=65.0,
        decision=Decision.HIGH_RISK,
        component_scores=component_scores(),
        reasons=["Configured high-risk threshold was exceeded"],
        model_version="model-v1",
        risk_config_version="risk-v2",
        feature_schema_version="features-v1",
    )
    assert result.decision is Decision.HIGH_RISK


def test_component_and_assessment_scores_reject_non_finite_values() -> None:
    with pytest.raises(ValidationError):
        ComponentScores(
            supervised_probability=0.2,
            anomaly_score=float("nan"),
            behavioral_risk=0.3,
            graph_risk=0.0,
        )
    with pytest.raises(ValidationError):
        FraudAssessment(
            application_id="APP-001",
            assessed_at=datetime(2026, 8, 19, 10, tzinfo=UTC),
            risk_score=float("nan"),
            decision=Decision.APPROVE,
            component_scores=component_scores(),
            model_version="model-v1",
            risk_config_version="risk-v1",
            feature_schema_version="features-v1",
        )
