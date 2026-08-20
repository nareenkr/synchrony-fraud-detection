"""Transactional repository for privacy-preserving fraud assessments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload, sessionmaker

from backend.app.core.privacy import Pseudonymizer, coarse_ip_network
from backend.app.explainability.reasons import MESSAGES, SOURCE_FALLBACK
from backend.app.schemas import (
    ComponentScores,
    Decision,
    FraudAssessment,
    InvestigatorOutcome,
    LendingChannel,
    LoanApplicationEvent,
)
from backend.app.schemas.assessments import RiskSignal

from .models import ApplicationRow, AssessmentRow, ReviewFeedbackRow, SignalRow

_NAMESPACE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class PersistedApplication:
    """Safe dashboard/detail projection; it deliberately omits all pseudonyms."""

    application_id: str
    event_timestamp: datetime
    assessed_at: datetime
    risk_score: float
    decision: Decision
    component_scores: ComponentScores
    signals: tuple[RiskSignal, ...]
    model_version: str
    risk_config_version: str
    feature_schema_version: str
    channel: LendingChannel

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(signal.message for signal in self.signals)


@dataclass(frozen=True, slots=True)
class AnalyticsSummary:
    total_applications: int
    approved_applications: int
    manual_reviews: int
    high_risk_applications: int
    average_risk_score: float
    fraud_high_risk_rate: float
    decision_counts: dict[str, int]

    @property
    def fraud_rate(self) -> float:
        """Dashboard-friendly alias for the high-risk decision rate."""

        return self.fraud_high_risk_rate


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    application_id: str
    outcome: InvestigatorOutcome
    reviewed_at: datetime


@dataclass(frozen=True, slots=True)
class LearningStatus:
    reviewed_applications: int
    confirmed_fraud: int
    legitimate: int
    inconclusive: int
    false_positive_reviews: int
    missed_fraud_reviews: int
    reviewed_alerts: int
    false_positive_review_rate: float
    minimum_feedback_required: int
    retraining_recommended: bool
    governance_status: str


class AssessmentRepository:
    """SQLAlchemy implementation of the decisioning ``AssessmentSink`` protocol."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        pseudonymizer: Pseudonymizer,
        *,
        namespace: str = "default",
    ) -> None:
        clean_namespace = namespace.strip().lower()
        if not _NAMESPACE_PATTERN.fullmatch(clean_namespace):
            raise ValueError("namespace must contain only lowercase letters, digits, '_' or '-'")
        self._sessions = session_factory
        self._pseudonymizer = pseudonymizer
        self.namespace = clean_namespace

    def close(self) -> None:
        """Release the repository engine and pooled database connections."""

        engine = self._sessions.kw.get("bind")
        if engine is not None:
            engine.dispose()

    def save(self, event: LoanApplicationEvent, assessment: FraudAssessment) -> None:
        """Insert or replace one application's assessment in a single transaction.

        Only identifier HMACs, coarse bands, scores, versions, and curated signal
        codes cross this boundary. In particular, raw income, requested amount,
        IP/device/bank/user identifiers, request bodies, and free-form reasons do not.
        """

        if event.application_id != assessment.application_id:
            raise ValueError("event and assessment application_id must match")
        session = self._sessions()
        try:
            with session.begin():
                application = session.scalar(
                    select(ApplicationRow)
                    .where(
                        ApplicationRow.namespace == self.namespace,
                        ApplicationRow.application_id == event.application_id,
                    )
                    .options(joinedload(ApplicationRow.assessment))
                )
                values = self._application_values(event)
                if application is None:
                    application = ApplicationRow(
                        namespace=self.namespace,
                        application_id=event.application_id,
                        **values,
                    )
                    session.add(application)
                else:
                    for field, value in values.items():
                        setattr(application, field, value)
                    if application.assessment is not None:
                        session.delete(application.assessment)
                        session.flush()

                stored = AssessmentRow(
                    assessed_at=assessment.assessed_at,
                    risk_score=assessment.risk_score,
                    decision=assessment.decision.value,
                    supervised_probability=assessment.component_scores.supervised_probability,
                    anomaly_score=assessment.component_scores.anomaly_score,
                    behavioral_risk=assessment.component_scores.behavioral_risk,
                    graph_risk=assessment.component_scores.graph_risk,
                    model_version=assessment.model_version,
                    risk_config_version=assessment.risk_config_version,
                    feature_schema_version=assessment.feature_schema_version,
                )
                stored.signals = [
                    SignalRow(position=index, code=code, source=source, severity=severity)
                    for index, (code, source, severity) in enumerate(
                        self._safe_signals(assessment.signals)
                    )
                ]
                application.assessment = stored
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_applications(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        decision: Decision | str | None = None,
    ) -> list[PersistedApplication]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        selected_decision = self._decision_value(decision)
        query = (
            select(ApplicationRow)
            .join(ApplicationRow.assessment)
            .where(ApplicationRow.namespace == self.namespace)
            .options(joinedload(ApplicationRow.assessment).joinedload(AssessmentRow.signals))
            .order_by(AssessmentRow.assessed_at.desc(), ApplicationRow.application_id.desc())
            .offset(offset)
            .limit(limit)
        )
        if selected_decision is not None:
            query = query.where(AssessmentRow.decision == selected_decision)
        with self._sessions() as session:
            rows = session.scalars(query).unique().all()
            return [self._project(row) for row in rows]

    def get_application(self, application_id: str) -> PersistedApplication | None:
        query = (
            select(ApplicationRow)
            .join(ApplicationRow.assessment)
            .where(
                ApplicationRow.namespace == self.namespace,
                ApplicationRow.application_id == application_id,
            )
            .options(joinedload(ApplicationRow.assessment).joinedload(AssessmentRow.signals))
        )
        with self._sessions() as session:
            row = session.scalar(query)
            return self._project(row) if row is not None else None

    def analytics(self) -> AnalyticsSummary:
        query = (
            select(
                AssessmentRow.decision,
                func.count(AssessmentRow.id),
                func.sum(AssessmentRow.risk_score),
            )
            .join(AssessmentRow.application)
            .where(ApplicationRow.namespace == self.namespace)
            .group_by(AssessmentRow.decision)
        )
        with self._sessions() as session:
            rows = session.execute(query).all()
        counts = {decision.value: 0 for decision in Decision}
        total_risk = 0.0
        for decision, count, risk_sum in rows:
            counts[str(decision)] = int(count)
            total_risk += float(risk_sum or 0.0)
        total = sum(counts.values())
        high_risk = counts[Decision.HIGH_RISK.value]
        return AnalyticsSummary(
            total_applications=total,
            approved_applications=counts[Decision.APPROVE.value],
            manual_reviews=counts[Decision.MANUAL_REVIEW.value],
            high_risk_applications=high_risk,
            average_risk_score=total_risk / total if total else 0.0,
            fraud_high_risk_rate=high_risk / total if total else 0.0,
            decision_counts=counts,
        )

    def record_review(
        self,
        application_id: str,
        outcome: InvestigatorOutcome | str,
        reviewer_id: str,
        *,
        reviewed_at: datetime | None = None,
    ) -> ReviewRecord | None:
        selected_outcome = InvestigatorOutcome(outcome)
        timestamp = reviewed_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("reviewed_at must be timezone-aware")
        session = self._sessions()
        try:
            with session.begin():
                application = session.scalar(
                    select(ApplicationRow)
                    .where(
                        ApplicationRow.namespace == self.namespace,
                        ApplicationRow.application_id == application_id,
                    )
                    .options(joinedload(ApplicationRow.review_feedback))
                )
                if application is None:
                    return None
                feedback = application.review_feedback
                if feedback is None:
                    feedback = ReviewFeedbackRow(application=application)
                    session.add(feedback)
                feedback.outcome = selected_outcome.value
                feedback.reviewed_at = timestamp
                feedback.reviewer_pseudonym = self._pseudonymizer.pseudonymize(
                    "reviewer", reviewer_id
                )
            return ReviewRecord(application_id, selected_outcome, timestamp.astimezone(UTC))
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def learning_status(self, *, minimum_feedback_required: int = 100) -> LearningStatus:
        query = (
            select(AssessmentRow.decision, ReviewFeedbackRow.outcome)
            .join(AssessmentRow.application)
            .join(ApplicationRow.review_feedback)
            .where(ApplicationRow.namespace == self.namespace)
        )
        with self._sessions() as session:
            rows = session.execute(query).all()
        counts = {outcome.value: 0 for outcome in InvestigatorOutcome}
        false_positives = 0
        missed_fraud = 0
        reviewed_alerts = 0
        for decision_value, outcome_value in rows:
            counts[str(outcome_value)] += 1
            alerted = decision_value in {
                Decision.MANUAL_REVIEW.value,
                Decision.HIGH_RISK.value,
            }
            if alerted:
                reviewed_alerts += 1
                if outcome_value == InvestigatorOutcome.LEGITIMATE.value:
                    false_positives += 1
            elif outcome_value == InvestigatorOutcome.CONFIRMED_FRAUD.value:
                missed_fraud += 1
        reviewed = len(rows)
        false_positive_rate = false_positives / reviewed_alerts if reviewed_alerts else 0.0
        enough_evidence = reviewed >= minimum_feedback_required
        recommendation = enough_evidence and (missed_fraud > 0 or false_positive_rate > 0.20)
        if not enough_evidence:
            governance_status = "COLLECTING_FEEDBACK"
        elif recommendation:
            governance_status = "RETRAINING_REVIEW_REQUIRED"
        else:
            governance_status = "MONITORING"
        return LearningStatus(
            reviewed_applications=reviewed,
            confirmed_fraud=counts[InvestigatorOutcome.CONFIRMED_FRAUD.value],
            legitimate=counts[InvestigatorOutcome.LEGITIMATE.value],
            inconclusive=counts[InvestigatorOutcome.INCONCLUSIVE.value],
            false_positive_reviews=false_positives,
            missed_fraud_reviews=missed_fraud,
            reviewed_alerts=reviewed_alerts,
            false_positive_review_rate=false_positive_rate,
            minimum_feedback_required=minimum_feedback_required,
            retraining_recommended=recommendation,
            governance_status=governance_status,
        )

    def ping(self) -> bool:
        try:
            with self._sessions() as session:
                session.execute(select(1)).scalar_one()
            return True
        except SQLAlchemyError:
            return False

    def reset_demo_namespace(self) -> int:
        """Delete only this repository's explicitly demo-scoped records."""

        is_demo = self.namespace == "demo" or self.namespace.startswith(("demo-", "demo_"))
        if not is_demo:
            raise PermissionError("only 'demo', 'demo-*', or 'demo_*' namespaces may be reset")
        session = self._sessions()
        try:
            with session.begin():
                result = session.execute(
                    delete(ApplicationRow).where(ApplicationRow.namespace == self.namespace)
                )
                return int(result.rowcount or 0)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _application_values(self, event: LoanApplicationEvent) -> dict[str, object]:
        return {
            "event_timestamp": event.event_timestamp,
            "user_pseudonym": self._pseudonymizer.pseudonymize("user", event.user_id),
            "device_pseudonym": self._pseudonym("device", event.device_id),
            "ip_network_pseudonym": self._pseudonym(
                "ip_network", coarse_ip_network(event.ip_address) if event.ip_address else None
            ),
            "bank_account_pseudonym": self._pseudonym("bank_account", event.bank_account_id),
            "account_age_band": self._account_age_band(event.account_age_days),
            "loan_to_income_band": self._loan_to_income_band(
                event.requested_loan_amount, event.income
            ),
            "channel": event.channel.value,
        }

    def _pseudonym(self, namespace: str, value: str | None) -> str | None:
        return self._pseudonymizer.pseudonymize(namespace, value) if value is not None else None

    @staticmethod
    def _safe_signals(signals: list[RiskSignal]) -> list[tuple[str, str, float]]:
        safe: list[tuple[str, str, float]] = []
        for signal in signals:
            code = signal.code
            if code not in MESSAGES:
                code = SOURCE_FALLBACK[signal.source][0]
            safe.append((code, signal.source, signal.severity))
        return safe

    @staticmethod
    def _decision_value(decision: Decision | str | None) -> str | None:
        if decision is None:
            return None
        return Decision(decision).value

    @staticmethod
    def _account_age_band(days: int | None) -> str:
        if days is None:
            return "UNKNOWN"
        if days <= 7:
            return "0_7_DAYS"
        if days <= 30:
            return "8_30_DAYS"
        if days <= 180:
            return "31_180_DAYS"
        if days <= 365:
            return "181_365_DAYS"
        return "366_PLUS_DAYS"

    @staticmethod
    def _loan_to_income_band(amount: float, income: float | None) -> str:
        if income is None:
            return "UNKNOWN"
        ratio = amount / income
        if ratio <= 0.25:
            return "UP_TO_0_25"
        if ratio <= 0.75:
            return "0_25_TO_0_75"
        if ratio <= 1.5:
            return "0_75_TO_1_5"
        return "ABOVE_1_5"

    @staticmethod
    def _project(row: ApplicationRow) -> PersistedApplication:
        assessment = row.assessment
        if assessment is None:  # guarded by the inner join; retained for type/runtime safety
            raise RuntimeError("application has no assessment")
        signals = tuple(
            RiskSignal(
                code=signal.code,
                message=MESSAGES.get(
                    signal.code, "A fraud detection component raised a risk signal."
                ),
                severity=signal.severity,
                source=signal.source,
            )
            for signal in assessment.signals
        )
        return PersistedApplication(
            application_id=row.application_id,
            event_timestamp=row.event_timestamp,
            assessed_at=assessment.assessed_at,
            risk_score=assessment.risk_score,
            decision=Decision(assessment.decision),
            component_scores=ComponentScores(
                supervised_probability=assessment.supervised_probability,
                anomaly_score=assessment.anomaly_score,
                behavioral_risk=assessment.behavioral_risk,
                graph_risk=assessment.graph_risk,
            ),
            signals=signals,
            model_version=assessment.model_version,
            risk_config_version=assessment.risk_config_version,
            feature_schema_version=assessment.feature_schema_version,
            channel=LendingChannel(row.channel),
        )
