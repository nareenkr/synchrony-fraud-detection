from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.api.dependencies import AppContainer
from backend.app.core.settings import Settings
from backend.app.main import create_app

READ_KEY = "reader-key-that-is-at-least-32-characters"
ADMIN_KEY = "admin-key-that-is-distinct-and-32-characters"


def protected_client() -> TestClient:
    settings = Settings(
        app_env="test",
        pseudonym_key="test-key-that-is-long-enough-for-validation",
        cors_origins=("https://dashboard.test",),
        auth_enabled=True,
        api_read_key=READ_KEY,
        api_admin_key=ADMIN_KEY,
    )
    settings.validate()
    return TestClient(
        create_app(
            settings=settings,
            container=AppContainer(settings=settings, startup_error="test-boundary"),
        )
    )


def test_health_is_public_but_data_routes_require_credentials() -> None:
    with protected_client() as client:
        assert client.get("/health").status_code == 200
        response = client.get("/model-info")
        assert response.status_code == 401
        assert response.json()["detail"] == "Valid API credentials are required"


def test_reader_can_read_but_cannot_mutate() -> None:
    with protected_client() as client:
        read = client.get("/model-info", headers={"X-API-Key": READ_KEY})
        assert read.status_code != 401
        write = client.post("/demo/reset", headers={"X-API-Key": READ_KEY})
        assert write.status_code == 403
        assert write.json()["detail"] == "Administrator access is required"


def test_admin_passes_authentication_and_authorization_gate() -> None:
    with protected_client() as client:
        response = client.post("/demo/reset", headers={"X-API-Key": ADMIN_KEY})
        assert response.status_code not in {401, 403}


def test_production_rejects_disabled_authentication() -> None:
    settings = Settings(
        app_env="production",
        pseudonym_key="production-key-that-is-long-enough-to-use",
        auth_enabled=False,
    )
    try:
        settings.validate()
    except ValueError as exc:
        assert "AUTH_ENABLED" in str(exc)
    else:
        raise AssertionError("Production settings accepted disabled authentication")
