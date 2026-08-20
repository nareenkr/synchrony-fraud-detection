"""Public HTTP response contracts.

The dashboard contracts deliberately expose only assessment outputs and safe
signals.  Pseudonymous database linkage fields remain an implementation detail
of the persistence adapter.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from backend.app.schemas import (
    ComponentScores,
    Decision,
    InvestigatorOutcome,
    LendingChannel,
    RiskSignal,
)
from backend.app.schemas.events import Identifier, LoanApplicationEvent


def _parse_json_datetime(value: Any) -> Any:
    """Parse the one representation JSON cannot natively carry.

    Every other canonical input remains strict.  This targeted conversion
    accepts ISO-8601 JSON timestamps without enabling broad coercion for
    numeric, boolean, or identifier fields.
    """

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value


def _parse_json_channel(value: Any) -> Any:
    """Convert the JSON string representation while keeping the domain model strict."""

    if isinstance(value, str):
        try:
            return LendingChannel(value)
        except ValueError:
            return value
    return value


class LoanApplicationRequest(LoanApplicationEvent):
    event_timestamp: Annotated[datetime, BeforeValidator(_parse_json_datetime)]
    channel: Annotated[LendingChannel, BeforeValidator(_parse_json_channel)] = LendingChannel.WEB


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class HealthCheck(_ResponseModel):
    status: str


class ReadinessCheck(_ResponseModel):
    status: str
    checks: dict[str, bool]


class HealthResponse(_ResponseModel):
    status: str
    liveness: HealthCheck
    readiness: ReadinessCheck


class ModelInfoResponse(_ResponseModel):
    model_version: str
    model_name: str
    feature_schema_version: str
    classifier_threshold: float = Field(ge=0, le=1)
    metrics: dict[str, float | int | None]
    prototype_only: bool
    risk_config_version: str | None = None
    thresholds: dict[str, float] | None = None
    weights: dict[str, float] | None = None
    features: list[dict[str, str | float]] = Field(default_factory=list)


class ApplicationSummary(_ResponseModel):
    application_id: Identifier
    event_timestamp: datetime
    assessed_at: datetime
    risk_score: float = Field(ge=0, le=100)
    decision: Decision
    component_scores: ComponentScores
    signals: list[RiskSignal] = Field(default_factory=list, max_length=50)
    model_version: str
    risk_config_version: str
    feature_schema_version: str
    channel: LendingChannel
    recommended_action: Literal[
        "CONTINUE_WITH_STANDARD_CHECKS",
        "REQUIRE_ENHANCED_VERIFICATION",
        "HOLD_AND_INVESTIGATE",
    ]

    @field_validator("event_timestamp", "assessed_at")
    @classmethod
    def timestamps_are_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("persisted timestamps must include a UTC offset")
        return value


class ApplicationDetail(ApplicationSummary):
    reasons: list[str] = Field(default_factory=list, max_length=20)


class AnalyticsResponse(_ResponseModel):
    total_applications: int = Field(ge=0)
    approved_applications: int = Field(ge=0)
    manual_reviews: int = Field(ge=0)
    high_risk_applications: int = Field(ge=0)
    average_risk_score: float = Field(ge=0, le=100)
    fraud_high_risk_rate: float = Field(ge=0, le=1)
    decision_counts: dict[str, int]

    @field_validator("decision_counts")
    @classmethod
    def valid_decision_counts(cls, value: dict[str, int]) -> dict[str, int]:
        expected = {item.value for item in Decision}
        if not set(value).issubset(expected) or any(count < 0 for count in value.values()):
            raise ValueError("decision_counts contains an invalid decision or count")
        return value


class DemoRunRequest(_ResponseModel):
    scenario: Literal["mixed", "normal", "suspicious", "fraud_ring"] = "mixed"
    interval_ms: int = Field(default=750, ge=50, le=60_000)
    repeat: int = Field(default=1, ge=1, le=10)


class RandomDemoRunRequest(_ResponseModel):
    count: int = Field(default=100, ge=1, le=5_000)
    interval_ms: int = Field(default=500, ge=50, le=60_000)
    seed: int = Field(default=20_260_820, ge=0, le=2_147_483_647)
    normal_percent: int = Field(default=80, ge=0, le=100)
    suspicious_percent: int = Field(default=15, ge=0, le=100)
    fraud_percent: int = Field(default=5, ge=0, le=100)

    @model_validator(mode="after")
    def profile_mix_totals_one_hundred(self) -> RandomDemoRunRequest:
        if self.normal_percent + self.suspicious_percent + self.fraud_percent != 100:
            raise ValueError("normal, suspicious, and fraud percentages must total 100")
        return self


class InvestigatorReviewRequest(_ResponseModel):
    outcome: InvestigatorOutcome
    reviewer_id: Identifier


class InvestigatorReviewResponse(_ResponseModel):
    application_id: Identifier
    outcome: InvestigatorOutcome
    reviewed_at: datetime


class LearningStatusResponse(_ResponseModel):
    reviewed_applications: int = Field(ge=0)
    confirmed_fraud: int = Field(ge=0)
    legitimate: int = Field(ge=0)
    inconclusive: int = Field(ge=0)
    false_positive_reviews: int = Field(ge=0)
    missed_fraud_reviews: int = Field(ge=0)
    reviewed_alerts: int = Field(ge=0)
    false_positive_review_rate: float = Field(ge=0, le=1)
    minimum_feedback_required: int = Field(ge=1)
    retraining_recommended: bool
    governance_status: Literal[
        "COLLECTING_FEEDBACK",
        "RETRAINING_REVIEW_REQUIRED",
        "MONITORING",
    ]


class SimulatorStatusResponse(_ResponseModel):
    running: bool
    scenario: str | None
    processed: int = Field(ge=0)
    total: int = Field(ge=0)
    last_application_id: str | None
    error: str | None


class ApiError(_ResponseModel):
    detail: str
    request_id: str | None = None


def public_record(record: Any, *, include_reasons: bool = False) -> dict[str, Any]:
    """Normalize an ORM/domain record without exposing persistence-only fields."""

    def value(name: str) -> Any:
        if isinstance(record, dict):
            return record[name]
        return getattr(record, name)

    signals = value("signals")
    decision = Decision(value("decision"))
    recommended_actions = {
        Decision.APPROVE: "CONTINUE_WITH_STANDARD_CHECKS",
        Decision.MANUAL_REVIEW: "REQUIRE_ENHANCED_VERIFICATION",
        Decision.HIGH_RISK: "HOLD_AND_INVESTIGATE",
    }
    payload = {
        "application_id": value("application_id"),
        "event_timestamp": value("event_timestamp"),
        "assessed_at": value("assessed_at"),
        "risk_score": value("risk_score"),
        "decision": decision,
        "component_scores": value("component_scores"),
        "signals": signals,
        "model_version": value("model_version"),
        "risk_config_version": value("risk_config_version"),
        "feature_schema_version": value("feature_schema_version"),
        "channel": value("channel"),
        "recommended_action": recommended_actions[decision],
    }
    if include_reasons:
        payload["reasons"] = [
            signal["message"] if isinstance(signal, dict) else signal.message for signal in signals
        ][:20]
    return payload
