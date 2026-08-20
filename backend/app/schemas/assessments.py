"""Fraud scoring and API response contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .events import Identifier


class Decision(StrEnum):
    APPROVE = "APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    HIGH_RISK = "HIGH_RISK"


class InvestigatorOutcome(StrEnum):
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    LEGITIMATE = "LEGITIMATE"
    INCONCLUSIVE = "INCONCLUSIVE"


ReasonCode = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=80, pattern=r"^[A-Z][A-Z0-9_]*$"
    ),
]


class ComponentScores(BaseModel):
    """Normalized outputs emitted by the four fraud detection components."""

    model_config = ConfigDict(extra="forbid", strict=True)

    supervised_probability: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    behavioral_risk: float = Field(ge=0, le=1)
    graph_risk: float = Field(ge=0, le=1)

    @field_validator("supervised_probability", "anomaly_score", "behavioral_risk", "graph_risk")
    @classmethod
    def finite_scores(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("component scores must be finite")
        return value


class RiskSignal(BaseModel):
    """Safe, display-ready evidence without raw SHAP values or identifiers."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    code: ReasonCode
    message: Annotated[str, StringConstraints(min_length=1, max_length=240)]
    severity: float = Field(ge=0, le=1)
    source: Annotated[
        str,
        StringConstraints(pattern=r"^(supervised|anomaly|behavioral|graph)$"),
    ]


class FraudAssessment(BaseModel):
    """Complete, internally consistent result returned from decisioning.

    Score/decision consistency is enforced by the configurable risk engine. It
    cannot be validated here without coupling this transport model to one
    particular configuration version.
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    application_id: Identifier
    assessed_at: datetime
    risk_score: float = Field(ge=0, le=100)
    decision: Decision
    component_scores: ComponentScores
    reasons: list[Annotated[str, StringConstraints(min_length=1, max_length=240)]] = Field(
        default_factory=list, max_length=20
    )
    signals: list[RiskSignal] = Field(default_factory=list, max_length=50)
    model_version: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    risk_config_version: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    feature_schema_version: Annotated[str, StringConstraints(min_length=1, max_length=80)]

    @field_validator("assessed_at")
    @classmethod
    def assessed_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("assessed_at must include a UTC offset")
        return value

    @field_validator("risk_score")
    @classmethod
    def finite_risk_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("risk_score must be finite")
        return value
