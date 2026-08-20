"""Configurable aggregation of normalized fraud-component outputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.schemas.assessments import ComponentScores, Decision


class RiskWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    supervised: float = Field(ge=0)
    anomaly: float = Field(ge=0)
    behavioral: float = Field(ge=0)
    graph: float = Field(ge=0)

    @model_validator(mode="after")
    def at_least_one_weight(self) -> RiskWeights:
        if sum(self.as_mapping().values()) <= 0:
            raise ValueError("at least one risk weight must be positive")
        return self

    def as_mapping(self) -> dict[str, float]:
        return {
            "supervised": self.supervised,
            "anomaly": self.anomaly,
            "behavioral": self.behavioral,
            "graph": self.graph,
        }


class RiskThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    manual_review: float = Field(ge=0, le=100)
    high_risk: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def ordered(self) -> RiskThresholds:
        if self.manual_review >= self.high_risk:
            raise ValueError("manual_review threshold must be below high_risk")
        return self


class RiskLabels(BaseModel):
    # YAML contains strings; enum coercion is intentional while unknown values
    # and extra keys remain rejected.
    model_config = ConfigDict(extra="forbid")

    approve: Decision
    manual_review: Decision
    high_risk: Decision

    @model_validator(mode="after")
    def canonical_labels(self) -> RiskLabels:
        expected = (Decision.APPROVE, Decision.MANUAL_REVIEW, Decision.HIGH_RISK)
        actual = (self.approve, self.manual_review, self.high_risk)
        if actual != expected:
            raise ValueError("risk labels must use the canonical decision ordering")
        return self


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    weights: RiskWeights
    thresholds: RiskThresholds
    labels: RiskLabels


@dataclass(frozen=True, slots=True)
class RiskDecision:
    risk_score: float
    decision: Decision
    config_version: str
    normalized_weights: Mapping[str, float]


def load_risk_config(path: str | Path) -> tuple[RiskConfig, str]:
    raw = Path(path).read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError("risk configuration must be a mapping")
    config = RiskConfig.model_validate(payload)
    version = f"risk-v{config.schema_version}-{hashlib.sha256(raw).hexdigest()[:12]}"
    return config, version


class RiskEngine:
    def __init__(self, config: RiskConfig, config_version: str) -> None:
        self.config = config
        self.config_version = config_version
        weights = config.weights.as_mapping()
        total = sum(weights.values())
        self.normalized_weights = {name: value / total for name, value in weights.items()}

    def combine(self, scores: ComponentScores) -> RiskDecision:
        components = {
            "supervised": scores.supervised_probability,
            "anomaly": scores.anomaly_score,
            "behavioral": scores.behavioral_risk,
            "graph": scores.graph_risk,
        }
        raw_score = 100.0 * sum(
            components[name] * weight for name, weight in self.normalized_weights.items()
        )
        risk_score = round(min(100.0, max(0.0, raw_score)), 2)
        decision = self.decision_for(risk_score)
        return RiskDecision(
            risk_score=risk_score,
            decision=decision,
            config_version=self.config_version,
            normalized_weights=dict(self.normalized_weights),
        )

    def decision_for(self, risk_score: float) -> Decision:
        if not 0 <= risk_score <= 100:
            raise ValueError("risk score must be between 0 and 100")
        if risk_score >= self.config.thresholds.high_risk:
            return self.config.labels.high_risk
        if risk_score >= self.config.thresholds.manual_review:
            return self.config.labels.manual_review
        return self.config.labels.approve
