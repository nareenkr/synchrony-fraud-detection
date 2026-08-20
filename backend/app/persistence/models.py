"""SQLAlchemy persistence models with an intentionally narrow data allowlist."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """Persist UTC and restore timezone awareness on SQLite and PostgreSQL."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("persisted timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class ApplicationRow(Base):
    """Privacy-reduced application envelope; raw request fields never enter this model."""

    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint("namespace", "application_id", name="uq_application_namespace_id"),
        Index("ix_applications_namespace_event", "namespace", "event_timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    namespace: Mapped[str] = mapped_column(String(64), nullable=False)
    application_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    user_pseudonym: Mapped[str] = mapped_column(String(64), nullable=False)
    device_pseudonym: Mapped[str | None] = mapped_column(String(64))
    ip_network_pseudonym: Mapped[str | None] = mapped_column(String(64))
    bank_account_pseudonym: Mapped[str | None] = mapped_column(String(64))
    account_age_band: Mapped[str] = mapped_column(String(24), nullable=False)
    loan_to_income_band: Mapped[str] = mapped_column(String(24), nullable=False)
    channel: Mapped[str] = mapped_column(String(24), nullable=False, default="WEB")

    assessment: Mapped[AssessmentRow] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    review_feedback: Mapped[ReviewFeedbackRow | None] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class AssessmentRow(Base):
    __tablename__ = "assessments"
    __table_args__ = (
        CheckConstraint("risk_score >= 0 AND risk_score <= 100", name="ck_risk_score"),
        CheckConstraint(
            "supervised_probability >= 0 AND supervised_probability <= 1",
            name="ck_supervised_probability",
        ),
        CheckConstraint("anomaly_score >= 0 AND anomaly_score <= 1", name="ck_anomaly_score"),
        CheckConstraint("behavioral_risk >= 0 AND behavioral_risk <= 1", name="ck_behavioral_risk"),
        CheckConstraint("graph_risk >= 0 AND graph_risk <= 1", name="ck_graph_risk"),
        Index("ix_assessments_assessed_at", "assessed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_row_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    assessed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    supervised_probability: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    behavioral_risk: Mapped[float] = mapped_column(Float, nullable=False)
    graph_risk: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    risk_config_version: Mapped[str] = mapped_column(String(80), nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(80), nullable=False)

    application: Mapped[ApplicationRow] = relationship(back_populates="assessment")
    signals: Mapped[list[SignalRow]] = relationship(
        back_populates="assessment",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SignalRow.position",
    )


class SignalRow(Base):
    __tablename__ = "signals"
    __table_args__ = (
        CheckConstraint("severity >= 0 AND severity <= 1", name="ck_signal_severity"),
        UniqueConstraint("assessment_id", "position", name="uq_signal_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assessment_id: Mapped[int] = mapped_column(
        ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)

    assessment: Mapped[AssessmentRow] = relationship(back_populates="signals")


class ReviewFeedbackRow(Base):
    """Pseudonymous investigator outcome used only for governed learning evidence."""

    __tablename__ = "review_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_row_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    outcome: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    reviewed_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    reviewer_pseudonym: Mapped[str] = mapped_column(String(64), nullable=False)

    application: Mapped[ApplicationRow] = relationship(back_populates="review_feedback")
