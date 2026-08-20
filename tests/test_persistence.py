from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy import inspect, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from backend.app.core.privacy import Pseudonymizer
from backend.app.persistence import (
    ApplicationRow,
    AssessmentRepository,
    AssessmentRow,
    Base,
    SignalRow,
    build_engine,
    build_session_factory,
    create_schema,
)
from backend.app.schemas import (
    ComponentScores,
    Decision,
    FraudAssessment,
    InvestigatorOutcome,
    LoanApplicationEvent,
    RiskSignal,
)

SECRET = "a-test-pseudonym-key-with-more-than-16-bytes"


def loan_event(
    application_id: str = "APP-100",
    *,
    at: datetime | None = None,
    user_id: str = "raw-user-8291",
) -> LoanApplicationEvent:
    return LoanApplicationEvent(
        application_id=application_id,
        user_id=user_id,
        event_timestamp=at
        or datetime(2026, 8, 20, 9, 15, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        requested_loan_amount=48_765.25,
        income=91_234.75,
        account_age_days=4,
        device_id="raw-device-secret-441",
        ip_address="198.51.100.77",
        bank_account_id="raw-bank-secret-772",
        geographic_region="private-region-name",
    )


def assessment(
    application_id: str = "APP-100",
    *,
    at: datetime | None = None,
    decision: Decision = Decision.MANUAL_REVIEW,
    risk_score: float = 62.0,
) -> FraudAssessment:
    return FraudAssessment(
        application_id=application_id,
        assessed_at=at or datetime(2026, 8, 20, 4, 0, tzinfo=UTC),
        risk_score=risk_score,
        decision=decision,
        component_scores=ComponentScores(
            supervised_probability=0.61,
            anomaly_score=0.45,
            behavioral_risk=0.72,
            graph_risk=0.35,
        ),
        reasons=["raw-user-8291 should never be stored in free-form explanation text"],
        signals=[
            RiskSignal(
                code="APPLICATION_VELOCITY",
                message="raw-device-secret-441 deliberately malicious message",
                severity=0.8,
                source="behavioral",
            ),
            RiskSignal(
                code="PRIVATE_IDENTIFIER_RAW_USER_8291",
                message="raw-bank-secret-772",
                severity=0.4,
                source="graph",
            ),
        ],
        model_version="classifier-v1",
        risk_config_version="risk-v2",
        feature_schema_version="features-v3",
    )


@pytest.fixture
def memory_repository() -> Iterator[AssessmentRepository]:
    engine = build_engine("sqlite:///:memory:")
    create_schema(engine)
    repository = AssessmentRepository(
        build_session_factory(engine), Pseudonymizer(SECRET), namespace="demo-test"
    )
    yield repository
    repository.close()


def test_save_get_round_trip_is_timezone_aware_and_privacy_reduced(
    memory_repository: AssessmentRepository,
) -> None:
    event = loan_event()
    expected = assessment()

    memory_repository.save(event, expected)
    stored = memory_repository.get_application(event.application_id)

    assert stored is not None
    assert stored.application_id == event.application_id
    assert stored.event_timestamp == event.event_timestamp.astimezone(UTC)
    assert stored.event_timestamp.tzinfo is UTC
    assert stored.assessed_at.tzinfo is UTC
    assert stored.risk_score == expected.risk_score
    assert stored.component_scores == expected.component_scores
    assert stored.decision is Decision.MANUAL_REVIEW
    assert [signal.code for signal in stored.signals] == [
        "APPLICATION_VELOCITY",
        "GRAPH_LINKAGE",
    ]
    assert all("raw-" not in reason for reason in stored.reasons)


def test_save_is_idempotent_per_namespace_and_replaces_assessment(
    memory_repository: AssessmentRepository,
) -> None:
    event = loan_event()
    memory_repository.save(event, assessment())
    replacement = assessment(
        at=datetime(2026, 8, 20, 5, 0, tzinfo=UTC),
        decision=Decision.HIGH_RISK,
        risk_score=91.0,
    )
    memory_repository.save(event, replacement)

    rows = memory_repository.list_applications()
    assert len(rows) == 1
    assert rows[0].risk_score == 91.0
    assert rows[0].decision is Decision.HIGH_RISK


def test_list_is_newest_first_paginated_bounded_and_filterable(
    memory_repository: AssessmentRepository,
) -> None:
    base = datetime(2026, 8, 20, tzinfo=UTC)
    cases = [
        ("APP-1", 1, Decision.APPROVE, 12.0),
        ("APP-2", 2, Decision.MANUAL_REVIEW, 55.0),
        ("APP-3", 3, Decision.HIGH_RISK, 88.0),
    ]
    for application_id, hour, decision, score in cases:
        memory_repository.save(
            loan_event(application_id, at=base + timedelta(hours=hour)),
            assessment(
                application_id,
                at=base + timedelta(hours=hour),
                decision=decision,
                risk_score=score,
            ),
        )

    assert [item.application_id for item in memory_repository.list_applications(limit=2)] == [
        "APP-3",
        "APP-2",
    ]
    assert memory_repository.list_applications(limit=1, offset=1)[0].application_id == "APP-2"
    assert memory_repository.list_applications(decision="HIGH_RISK")[0].application_id == "APP-3"
    with pytest.raises(ValueError, match="limit"):
        memory_repository.list_applications(limit=201)
    with pytest.raises(ValueError, match="offset"):
        memory_repository.list_applications(offset=-1)
    with pytest.raises(ValueError):
        memory_repository.list_applications(decision="DROP TABLE applications")


def test_analytics_returns_kpis_decision_counts_average_and_high_risk_rate(
    memory_repository: AssessmentRepository,
) -> None:
    for index, (decision, score) in enumerate(
        [
            (Decision.APPROVE, 10.0),
            (Decision.APPROVE, 20.0),
            (Decision.MANUAL_REVIEW, 50.0),
            (Decision.HIGH_RISK, 80.0),
        ]
    ):
        application_id = f"APP-{index}"
        memory_repository.save(
            loan_event(application_id),
            assessment(application_id, decision=decision, risk_score=score),
        )

    result = memory_repository.analytics()
    assert result.total_applications == 4
    assert result.approved_applications == 2
    assert result.manual_reviews == 1
    assert result.high_risk_applications == 1
    assert result.average_risk_score == pytest.approx(40.0)
    assert result.fraud_high_risk_rate == pytest.approx(0.25)
    assert result.decision_counts == {"APPROVE": 2, "MANUAL_REVIEW": 1, "HIGH_RISK": 1}


def test_empty_analytics_and_ping(memory_repository: AssessmentRepository) -> None:
    result = memory_repository.analytics()
    assert result.total_applications == 0
    assert result.average_risk_score == 0.0
    assert result.fraud_high_risk_rate == 0.0
    assert memory_repository.ping() is True


def test_investigator_feedback_drives_governed_learning_status(
    memory_repository: AssessmentRepository,
) -> None:
    cases = [
        ("APP-FP", Decision.HIGH_RISK, 90.0, InvestigatorOutcome.LEGITIMATE),
        ("APP-MISS", Decision.APPROVE, 10.0, InvestigatorOutcome.CONFIRMED_FRAUD),
        ("APP-UNCLEAR", Decision.MANUAL_REVIEW, 55.0, InvestigatorOutcome.INCONCLUSIVE),
    ]
    for application_id, decision, score, outcome in cases:
        memory_repository.save(
            loan_event(application_id),
            assessment(application_id, decision=decision, risk_score=score),
        )
        result = memory_repository.record_review(application_id, outcome, "raw-reviewer-007")
        assert result is not None
        assert result.outcome is outcome

    status = memory_repository.learning_status(minimum_feedback_required=3)
    assert status.reviewed_applications == 3
    assert status.confirmed_fraud == 1
    assert status.legitimate == 1
    assert status.inconclusive == 1
    assert status.false_positive_reviews == 1
    assert status.missed_fraud_reviews == 1
    assert status.reviewed_alerts == 2
    assert status.false_positive_review_rate == pytest.approx(0.5)
    assert status.retraining_recommended is True
    assert status.governance_status == "RETRAINING_REVIEW_REQUIRED"

    assert memory_repository.record_review("MISSING", outcome, "reviewer") is None


def test_demo_reset_is_exactly_namespaced_and_cascades() -> None:
    engine = build_engine("sqlite:///:memory:")
    create_schema(engine)
    factory = build_session_factory(engine)
    demo_a = AssessmentRepository(factory, Pseudonymizer(SECRET), namespace="demo-a")
    demo_b = AssessmentRepository(factory, Pseudonymizer(SECRET), namespace="demo-b")
    production = AssessmentRepository(factory, Pseudonymizer(SECRET), namespace="production")
    lookalike = AssessmentRepository(factory, Pseudonymizer(SECRET), namespace="demolition")
    demo_a.save(loan_event("APP-SAME"), assessment("APP-SAME"))
    demo_b.save(loan_event("APP-SAME"), assessment("APP-SAME"))
    production.save(loan_event("APP-PROD"), assessment("APP-PROD"))

    assert demo_a.reset_demo_namespace() == 1
    assert demo_a.get_application("APP-SAME") is None
    assert demo_b.get_application("APP-SAME") is not None
    assert production.get_application("APP-PROD") is not None
    with pytest.raises(PermissionError, match="demo"):
        production.reset_demo_namespace()
    with pytest.raises(PermissionError, match="demo"):
        lookalike.reset_demo_namespace()

    with factory() as session:
        remaining_signal = session.scalar(
            select(SignalRow)
            .join(SignalRow.assessment)
            .join(AssessmentRow.application)
            .where(ApplicationRow.namespace == "demo-a")
        )
        assert remaining_signal is None
    engine.dispose()


def test_file_database_contains_no_raw_identifiers_or_financial_values(tmp_path: Path) -> None:
    database_path = tmp_path / "privacy.sqlite3"
    engine = build_engine(f"sqlite:///{database_path.as_posix()}")
    create_schema(engine)
    factory = build_session_factory(engine)
    repository = AssessmentRepository(factory, Pseudonymizer(SECRET), namespace="demo-privacy")
    raw_event = loan_event()
    repository.save(raw_event, assessment())

    forbidden_columns = {
        "user_id",
        "device_id",
        "ip_address",
        "bank_account_id",
        "income",
        "requested_loan_amount",
        "request_body",
        "geographic_region",
        "message",
        "reason",
    }
    inspector = inspect(engine)
    actual_columns = {
        column["name"]
        for table in ("applications", "assessments", "signals")
        for column in inspector.get_columns(table)
    }
    assert forbidden_columns.isdisjoint(actual_columns)

    with engine.connect() as connection:
        values = connection.execute(
            text(
                "SELECT user_pseudonym, device_pseudonym, ip_network_pseudonym, "
                "bank_account_pseudonym, account_age_band, loan_to_income_band "
                "FROM applications"
            )
        ).one()
        signal_values = connection.execute(text("SELECT code, source FROM signals")).all()
    serialized = repr((values, signal_values))
    for raw_value in (
        raw_event.user_id,
        raw_event.device_id,
        raw_event.ip_address,
        raw_event.bank_account_id,
        raw_event.geographic_region,
        str(raw_event.income),
        str(raw_event.requested_loan_amount),
    ):
        assert raw_value not in serialized
        assert raw_value.encode() not in database_path.read_bytes()
    assert values.user_pseudonym.startswith("user_")
    assert values.device_pseudonym.startswith("device_")
    assert values.ip_network_pseudonym.startswith("ip_network_")
    engine.dispose()


def test_failed_flush_rolls_back_the_whole_save(memory_repository: AssessmentRepository) -> None:
    factory = memory_repository._sessions  # transaction behavior, not public state

    def fail_once(session: Session, flush_context: object, instances: object) -> None:
        del session, flush_context, instances
        raise RuntimeError("simulated database failure")

    sqlalchemy_event.listen(Session, "before_flush", fail_once)
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            memory_repository.save(loan_event("APP-ROLLBACK"), assessment("APP-ROLLBACK"))
    finally:
        sqlalchemy_event.remove(Session, "before_flush", fail_once)

    with factory() as session:
        assert (
            session.scalar(
                select(ApplicationRow).where(ApplicationRow.application_id == "APP-ROLLBACK")
            )
            is None
        )


def test_models_compile_for_postgresql_and_schema_creation_is_idempotent() -> None:
    dialect = postgresql.dialect()
    for table in Base.metadata.sorted_tables:
        assert str(CreateTable(table).compile(dialect=dialect))

    engine = build_engine("sqlite:///:memory:")
    create_schema(engine)
    create_schema(engine)
    assert set(inspect(engine).get_table_names()) == {
        "applications",
        "assessments",
        "review_feedback",
        "signals",
    }
    engine.dispose()
