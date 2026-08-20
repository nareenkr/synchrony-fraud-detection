"""FastAPI application factory and production dependency bootstrap."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import AppContainer, router
from backend.app.core.settings import Settings

logger = logging.getLogger(__name__)

ContainerFactory = Callable[[Settings], AppContainer]
MAX_REQUEST_BYTES = 64 * 1024
AUTH_EXEMPT_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


def build_default_container(settings: Settings) -> AppContainer:
    """Construct the real local runtime.

    Imports stay inside the function so importing the ASGI application has no
    database or model-loading side effects.  This also gives tests a clean
    dependency-injection seam.
    """

    from backend.app.core.privacy import Pseudonymizer
    from backend.app.fraud.risk_engine import RiskEngine, load_risk_config
    from backend.app.persistence import (
        AssessmentRepository,
        build_engine,
        build_session_factory,
        create_schema,
    )
    from backend.app.services.decisioning import DecisioningService
    from backend.app.services.model_registry import ModelInfoService, ModelRegistry
    from backend.app.services.simulator import DemoSimulator
    from backend.app.state import MemoryRealtimeStateStore, RedisRealtimeStateStore

    if settings.state_backend == "memory":
        state_store = MemoryRealtimeStateStore()
    elif settings.state_backend == "redis":
        state_store = RedisRealtimeStateStore.from_url(
            settings.redis_url,
            identifier_secret=settings.pseudonym_key,
            namespace=settings.persistence_namespace,
        )
        if not state_store.ping():
            raise RuntimeError("Redis state backend is unavailable")
    else:  # Settings validation makes this defensive.
        raise RuntimeError("The configured real-time state backend is unavailable")

    engine = build_engine(settings.database_url)
    create_schema(engine)
    repository = AssessmentRepository(
        build_session_factory(engine),
        Pseudonymizer(settings.pseudonym_key),
        namespace=settings.persistence_namespace,
    )
    registry = ModelRegistry.load(
        settings.model_bundle_path,
        settings.anomaly_artifact_path,
    )
    risk_config, risk_version = load_risk_config(settings.risk_config_path)
    risk_engine = RiskEngine(risk_config, risk_version)
    decisioning = DecisioningService(
        state_store=state_store,
        supervised=registry.supervised,
        anomaly=registry.anomaly,
        risk_engine=risk_engine,
        sink=repository,
    )
    simulator = DemoSimulator(decisioning, resetter=repository)
    return AppContainer(
        settings=settings,
        decisioning=decisioning,
        models=ModelInfoService(registry, risk_engine),
        repository=repository,
        simulator=simulator,
    )


def create_app(
    *,
    settings: Settings | None = None,
    container: AppContainer | None = None,
    container_factory: ContainerFactory = build_default_container,
) -> FastAPI:
    """Create an isolated application instance suitable for ASGI or tests."""

    resolved_settings = settings or (container.settings if container else Settings.from_env())

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if container is not None:
            application.state.container = container
        else:
            try:
                application.state.container = container_factory(resolved_settings)
            except Exception as exc:
                # An unavailable model or database must not make liveness
                # unknowable.  Store only the exception class, never its text,
                # because dependency messages may contain connection secrets.
                logger.error("Application startup dependency failed: %s", type(exc).__name__)
                application.state.container = AppContainer(
                    settings=resolved_settings,
                    startup_error=type(exc).__name__,
                )
        try:
            yield
        finally:
            active_container = getattr(application.state, "container", None)
            if isinstance(active_container, AppContainer):
                active_container.shutdown()

    application = FastAPI(
        title="Synchrony Fraud Decisioning API",
        version="0.1.0",
        description=(
            "Prototype decision-support API. Outputs require human oversight and are not suitable "
            "for autonomous lending decisions."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type", "X-API-Key", "X-Request-ID"],
    )

    @application.middleware("http")
    async def request_guard(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid4().hex
        if (
            resolved_settings.auth_enabled
            and request.method != "OPTIONS"
            and request.url.path not in AUTH_EXEMPT_PATHS
        ):
            supplied_key = request.headers.get("x-api-key", "")
            is_admin = secrets.compare_digest(supplied_key, resolved_settings.api_admin_key)
            is_reader = secrets.compare_digest(supplied_key, resolved_settings.api_read_key)
            if not (is_admin or is_reader):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "detail": "Valid API credentials are required",
                        "request_id": request_id,
                    },
                    headers={"X-Request-ID": request_id},
                )
            if request.method not in {"GET", "HEAD"} and not is_admin:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": "Administrator access is required",
                        "request_id": request_id,
                    },
                    headers={"X-Request-ID": request_id},
                )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > MAX_REQUEST_BYTES
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid Content-Length header", "request_id": request_id},
                    headers={"X-Request-ID": request_id},
                )
            if too_large:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": "Request body is too large", "request_id": request_id},
                    headers={"X-Request-ID": request_id},
                )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI's default response includes rejected input values.  Omit
        # those values so malformed requests cannot reflect financial or
        # identifying data into logs/UI error telemetry.
        request_id = request.headers.get("x-request-id") or uuid4().hex
        safe_errors = [
            {"type": item["type"], "loc": list(item["loc"]), "msg": item["msg"]}
            for item in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"detail": safe_errors, "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or uuid4().hex
        logger.error(
            "Unhandled API error request_id=%s method=%s path=%s type=%s",
            request_id,
            request.method,
            request.url.path,
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

    application.include_router(router)
    return application


app = create_app()
