"""Dependency-injection container and narrow API-facing protocols."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, runtime_checkable

from fastapi import Depends, HTTPException, Request, status

from backend.app.core.settings import Settings
from backend.app.schemas import Decision, FraudAssessment, LoanApplicationEvent


@runtime_checkable
class DecisioningProvider(Protocol):
    def assess(self, event: LoanApplicationEvent) -> FraudAssessment: ...


@runtime_checkable
class ModelInfoProvider(Protocol):
    def safe_info(self) -> dict[str, object]: ...


@runtime_checkable
class AssessmentRepository(Protocol):
    """Structural boundary implemented by the SQLAlchemy repository.

    Return values intentionally remain domain/ORM-neutral.  HTTP response
    conversion happens at the route boundary and whitelists public fields.
    """

    def save(self, event: LoanApplicationEvent, assessment: FraudAssessment) -> None: ...

    def list_applications(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        decision: Decision | str | None = None,
    ) -> Sequence[Any]: ...

    def get_application(self, application_id: str) -> Any | None: ...

    def analytics(self) -> Any: ...

    def record_review(self, application_id: str, outcome: str, reviewer_id: str) -> Any: ...

    def learning_status(self) -> Any: ...

    def ping(self) -> bool: ...


@runtime_checkable
class SimulatorProvider(Protocol):
    def start(self, scenario: str = "mixed", *, interval_ms: int = 750, repeat: int = 1) -> Any: ...

    def start_random(
        self,
        *,
        count: int = 100,
        interval_ms: int = 500,
        seed: int = 20_260_820,
        normal_percent: int = 80,
        suspicious_percent: int = 15,
        fraud_percent: int = 5,
    ) -> Any: ...

    def stop(self) -> Any: ...

    def reset(self) -> Any: ...

    def status(self) -> Any: ...


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    decisioning: DecisioningProvider | None = None
    models: ModelInfoProvider | None = None
    repository: AssessmentRepository | None = None
    simulator: SimulatorProvider | None = None
    startup_error: str | None = None

    def readiness(self) -> dict[str, bool]:
        repository_ready = False
        if self.repository is not None:
            try:
                repository_ready = bool(self.repository.ping())
            except Exception:
                repository_ready = False
        state_ready = False
        if self.decisioning is not None:
            state_store = getattr(self.decisioning, "state_store", None)
            if state_store is not None:
                ping = getattr(state_store, "ping", None)
                try:
                    state_ready = bool(ping()) if callable(ping) else True
                except Exception:
                    state_ready = False
        return {
            "models": self.models is not None and self.decisioning is not None,
            "persistence": repository_ready,
            "state": state_ready,
            "startup": self.startup_error is None,
        }

    def shutdown(self) -> None:
        """Stop background work and release external resources once."""

        if self.simulator is not None:
            self.simulator.stop()
        state_store = (
            getattr(self.decisioning, "state_store", None)
            if self.decisioning is not None
            else None
        )
        for resource in (state_store, self.repository):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


def get_container(request: Request) -> AppContainer:
    container = getattr(request.app.state, "container", None)
    if not isinstance(container, AppContainer):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready",
        )
    return container


def get_decisioning(
    container: Annotated[AppContainer, Depends(get_container)],
) -> DecisioningProvider:
    if container.decisioning is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Decisioning service is not ready",
        )
    return container.decisioning


def get_models(
    container: Annotated[AppContainer, Depends(get_container)],
) -> ModelInfoProvider:
    if container.models is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not ready",
        )
    return container.models


def get_repository(
    container: Annotated[AppContainer, Depends(get_container)],
) -> AssessmentRepository:
    if container.repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistence is not ready",
        )
    return container.repository


def get_simulator(
    container: Annotated[AppContainer, Depends(get_container)],
) -> SimulatorProvider:
    if container.simulator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demo simulator is not ready",
        )
    return container.simulator
