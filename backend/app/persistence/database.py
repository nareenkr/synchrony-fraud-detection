"""Database construction and schema lifecycle helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base

SessionFactory = sessionmaker[Session]


def build_engine(database_url: str = "sqlite:///./synchrony.db", **kwargs: Any) -> Engine:
    """Build a SQLAlchemy engine suitable for SQLite or PostgreSQL.

    A shared pool makes ``sqlite:///:memory:`` useful across repository sessions.
    PostgreSQL URLs are passed through without dialect-specific model changes.
    """

    options = dict(kwargs)
    if database_url in {"sqlite://", "sqlite:///:memory:"}:
        options.setdefault("poolclass", StaticPool)
        options.setdefault("connect_args", {"check_same_thread": False})
    elif database_url.startswith("sqlite:"):
        options.setdefault("connect_args", {"check_same_thread": False})
    engine = create_engine(database_url, **options)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def build_session_factory(engine: Engine) -> SessionFactory:
    """Return short-lived, non-expiring ORM sessions bound to ``engine``."""

    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def create_schema(engine: Engine) -> None:
    """Create the prototype schema (migrations replace this in production)."""

    Base.metadata.create_all(engine)
    columns = {column["name"] for column in inspect(engine).get_columns("applications")}
    if "channel" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE applications ADD COLUMN channel "
                    "VARCHAR(24) NOT NULL DEFAULT 'WEB'"
                )
            )


@contextmanager
def session_scope(factory: SessionFactory) -> Iterator[Session]:
    """Provide an explicit commit/rollback boundary for callers needing a session."""

    session = factory()
    try:
        with session.begin():
            yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
