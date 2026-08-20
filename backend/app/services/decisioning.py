"""End-to-end orchestration for one real-time fraud assessment."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

import pandas as pd

from backend.app.explainability import ExplanationService
from backend.app.features import FEATURE_SCHEMA_VERSION, FeatureBuilder
from backend.app.fraud.anomaly import AnomalyScorer
from backend.app.fraud.behavioral import BehavioralScorer
from backend.app.fraud.graph import GraphScorer
from backend.app.fraud.risk_engine import RiskEngine
from backend.app.fraud.supervised import SupervisedModel
from backend.app.schemas import (
    ComponentScores,
    FraudAssessment,
    LoanApplicationEvent,
    RiskSignal,
)
from backend.app.state import RealtimeStateStore


class AssessmentSink(Protocol):
    """Persistence boundary; the SQL repository implements this in M6."""

    def save(self, event: LoanApplicationEvent, assessment: FraudAssessment) -> None: ...


class DecisioningService:
    """Combine point-in-time state, all scorers, policy, and explanations."""

    def __init__(
        self,
        *,
        state_store: RealtimeStateStore,
        supervised: SupervisedModel,
        anomaly: AnomalyScorer,
        risk_engine: RiskEngine,
        feature_builder: FeatureBuilder | None = None,
        behavioral: BehavioralScorer | None = None,
        graph: GraphScorer | None = None,
        explanations: ExplanationService | None = None,
        sink: AssessmentSink | None = None,
    ) -> None:
        self.state_store = state_store
        self.supervised = supervised
        self.anomaly = anomaly
        self.risk_engine = risk_engine
        self.feature_builder = feature_builder or FeatureBuilder()
        self.behavioral = behavioral or BehavioralScorer()
        self.graph = graph or GraphScorer()
        self.explanations = explanations or ExplanationService()
        self.sink = sink

    def assess(
        self,
        event: LoanApplicationEvent,
        *,
        now: datetime | None = None,
    ) -> FraudAssessment:
        """Assess and record one event using a prior-only atomic snapshot.

        The state adapter atomically obtains the prior snapshot and records the
        current event, preventing concurrent applications from receiving the
        same velocity position. Model readiness is validated during service
        construction/startup, so scoring failures are exceptional.
        """

        snapshot = self.state_store.snapshot_and_record(event, now=now)
        features = self.feature_builder.transform(event, snapshot)
        supervised_probability = float(self.supervised.predict_proba(features)[0])
        anomaly_score = float(self.anomaly.score(features))
        behavioral = self.behavioral.score(features)
        graph = self.graph.score(features)
        components = ComponentScores(
            supervised_probability=supervised_probability,
            anomaly_score=anomaly_score,
            behavioral_risk=float(behavioral.risk),
            graph_risk=float(graph.risk),
        )
        risk = self.risk_engine.combine(components)

        component_signals: list[RiskSignal] = [*behavioral.signals, *graph.signals]
        if anomaly_score >= 0.75:
            component_signals.append(
                RiskSignal(
                    code="ANOMALOUS_PATTERN",
                    message="Application behavior is unusual compared with prior normal activity.",
                    severity=anomaly_score,
                    source="anomaly",
                )
            )
        explanation = self.explanations.explain(
            features=features,
            model=self.supervised.estimator,
            component_signals=component_signals,
        )
        assessed_at = now or datetime.now(UTC)
        if assessed_at.tzinfo is None or assessed_at.utcoffset() is None:
            raise ValueError("assessment time must be timezone-aware")
        assessment = FraudAssessment(
            application_id=event.application_id,
            assessed_at=assessed_at,
            risk_score=risk.risk_score,
            decision=risk.decision,
            component_scores=components,
            reasons=explanation.reasons,
            signals=explanation.signals,
            model_version=self.supervised.model_version,
            risk_config_version=risk.config_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
        if self.sink is not None:
            self.sink.save(event, assessment)
        return assessment

    def score_features(self, event: LoanApplicationEvent) -> pd.DataFrame:
        """Read-only feature preview for tests and internal investigation."""

        snapshot = self.state_store.get_snapshot(event)
        return self.feature_builder.transform(event, snapshot)
