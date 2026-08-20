"""HTTP routes for scoring, model metadata, and dashboard queries."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from backend.app.schemas import Decision, FraudAssessment

from .dependencies import (
    AppContainer,
    AssessmentRepository,
    DecisioningProvider,
    ModelInfoProvider,
    SimulatorProvider,
    get_container,
    get_decisioning,
    get_models,
    get_repository,
    get_simulator,
)
from .schemas import (
    AnalyticsResponse,
    ApplicationDetail,
    ApplicationSummary,
    DemoRunRequest,
    HealthCheck,
    HealthResponse,
    InvestigatorReviewRequest,
    InvestigatorReviewResponse,
    LearningStatusResponse,
    LoanApplicationRequest,
    ModelInfoResponse,
    RandomDemoRunRequest,
    ReadinessCheck,
    SimulatorStatusResponse,
    public_record,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["operations"])
def health(container: Annotated[AppContainer, Depends(get_container)]) -> HealthResponse:
    checks = container.readiness()
    ready = all(checks.values())
    return HealthResponse(
        status="ok" if ready else "degraded",
        liveness=HealthCheck(status="alive"),
        readiness=ReadinessCheck(
            status="ready" if ready else "not_ready",
            checks=checks,
        ),
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["operations"])
def model_info(models: Annotated[ModelInfoProvider, Depends(get_models)]) -> ModelInfoResponse:
    return ModelInfoResponse.model_validate(models.safe_info())


@router.post(
    "/predict",
    response_model=FraudAssessment,
    status_code=status.HTTP_200_OK,
    tags=["decisioning"],
)
def predict(
    event: LoanApplicationRequest,
    decisioning: Annotated[DecisioningProvider, Depends(get_decisioning)],
) -> FraudAssessment:
    # Do not log or attach the raw request to errors: the event can contain
    # financial, device, IP, and location observations.
    return decisioning.assess(event)


@router.get(
    "/applications",
    response_model=list[ApplicationSummary],
    tags=["dashboard"],
)
def applications(
    repository: Annotated[AssessmentRepository, Depends(get_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=1_000_000)] = 0,
    decision: Decision | None = None,
) -> list[ApplicationSummary]:
    records = repository.list_applications(limit=limit, offset=offset, decision=decision)
    return [ApplicationSummary.model_validate(public_record(record)) for record in records]


@router.get(
    "/applications/{application_id}",
    response_model=ApplicationDetail,
    responses={404: {"description": "Application not found"}},
    tags=["dashboard"],
)
def application_detail(
    application_id: Annotated[
        str,
        Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]*$"),
    ],
    repository: Annotated[AssessmentRepository, Depends(get_repository)],
) -> ApplicationDetail:
    record = repository.get_application(application_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    return ApplicationDetail.model_validate(public_record(record, include_reasons=True))


@router.post(
    "/applications/{application_id}/review",
    response_model=InvestigatorReviewResponse,
    responses={404: {"description": "Application not found"}},
    tags=["learning"],
)
def record_investigator_review(
    application_id: Annotated[
        str,
        Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]*$"),
    ],
    request: InvestigatorReviewRequest,
    repository: Annotated[AssessmentRepository, Depends(get_repository)],
) -> InvestigatorReviewResponse:
    record = repository.record_review(application_id, request.outcome, request.reviewer_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    return InvestigatorReviewResponse.model_validate(record)


@router.get("/learning/status", response_model=LearningStatusResponse, tags=["learning"])
def learning_status(
    repository: Annotated[AssessmentRepository, Depends(get_repository)],
) -> LearningStatusResponse:
    return LearningStatusResponse.model_validate(repository.learning_status())


@router.get("/analytics", response_model=AnalyticsResponse, tags=["dashboard"])
def analytics(
    repository: Annotated[AssessmentRepository, Depends(get_repository)],
) -> AnalyticsResponse:
    return AnalyticsResponse.model_validate(repository.analytics())


@router.post("/demo/run", response_model=SimulatorStatusResponse, tags=["demo"])
def run_demo(
    request: DemoRunRequest,
    simulator: Annotated[SimulatorProvider, Depends(get_simulator)],
) -> SimulatorStatusResponse:
    try:
        status_value = simulator.start(
            request.scenario,
            interval_ms=request.interval_ms,
            repeat=request.repeat,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return SimulatorStatusResponse.model_validate(status_value)


@router.post("/demo/random/run", response_model=SimulatorStatusResponse, tags=["demo"])
def run_random_demo(
    request: RandomDemoRunRequest,
    simulator: Annotated[SimulatorProvider, Depends(get_simulator)],
) -> SimulatorStatusResponse:
    try:
        status_value = simulator.start_random(**request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return SimulatorStatusResponse.model_validate(status_value)


@router.post("/demo/stop", response_model=SimulatorStatusResponse, tags=["demo"])
def stop_demo(
    simulator: Annotated[SimulatorProvider, Depends(get_simulator)],
) -> SimulatorStatusResponse:
    return SimulatorStatusResponse.model_validate(simulator.stop())


@router.post("/demo/reset", response_model=SimulatorStatusResponse, tags=["demo"])
def reset_demo(
    simulator: Annotated[SimulatorProvider, Depends(get_simulator)],
) -> SimulatorStatusResponse:
    return SimulatorStatusResponse.model_validate(simulator.reset())


@router.get("/demo/status", response_model=SimulatorStatusResponse, tags=["demo"])
def demo_status(
    simulator: Annotated[SimulatorProvider, Depends(get_simulator)],
) -> SimulatorStatusResponse:
    return SimulatorStatusResponse.model_validate(simulator.status())
