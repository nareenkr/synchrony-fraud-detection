from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.api.dependencies import AppContainer, get_decisioning
from backend.app.core.settings import Settings
from backend.app.fraud.risk_engine import RiskEngine, load_risk_config
from backend.app.fraud.supervised import SupervisedModel
from backend.app.main import create_app
from backend.app.schemas import Decision, FraudAssessment, InvestigatorOutcome, LoanApplicationEvent
from backend.app.services.decisioning import DecisioningService
from backend.app.services.simulator import DemoSimulator
from backend.app.state import MemoryRealtimeStateStore


class RatioEstimator:
    def predict_proba(self, frame: Any) -> np.ndarray:
        probability = np.clip(
            0.04
            + 0.34 * frame["loan_to_income_ratio"].to_numpy()
            + 0.04 * frame["applications_last_hour"].to_numpy()
            + 0.03 * frame["shared_device_user_count"].to_numpy(),
            0.0,
            1.0,
        )
        return np.column_stack((1.0 - probability, probability))


class FixedAnomaly:
    def score(self, _features: Any) -> float:
        return 0.1


class FakeRegistry:
    def safe_info(self) -> dict[str, object]:
        return {
            "model_version": "api-test-v1",
            "model_name": "ratio-test",
            "feature_schema_version": "loan-features-v1",
            "classifier_threshold": 0.42,
            "metrics": {
                "precision": 0.8,
                "recall": 0.9,
                "f1": 0.85,
                "roc_auc": 0.92,
                "pr_auc": 0.77,
                "false_positive_rate": 0.05,
            },
            "prototype_only": True,
        }


class FakeRepository:
    def __init__(self) -> None:
        self.records: list[SimpleNamespace] = []
        self.reviews: dict[str, InvestigatorOutcome] = {}
        self.available = True

    def save(self, event: LoanApplicationEvent, assessment: FraudAssessment) -> None:
        self.records.append(
            SimpleNamespace(
                application_id=assessment.application_id,
                event_timestamp=event.event_timestamp,
                assessed_at=assessment.assessed_at,
                risk_score=assessment.risk_score,
                decision=assessment.decision,
                component_scores=assessment.component_scores,
                signals=assessment.signals,
                model_version=assessment.model_version,
                risk_config_version=assessment.risk_config_version,
                feature_schema_version=assessment.feature_schema_version,
                channel=event.channel,
            )
        )

    def list_applications(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        decision: Decision | str | None = None,
    ) -> list[SimpleNamespace]:
        expected = Decision(decision) if decision is not None else None
        records = [
            item for item in reversed(self.records) if expected is None or item.decision is expected
        ]
        return records[offset : offset + limit]

    def get_application(self, application_id: str) -> SimpleNamespace | None:
        return next(
            (item for item in reversed(self.records) if item.application_id == application_id),
            None,
        )

    def analytics(self) -> SimpleNamespace:
        counts = Counter(item.decision.value for item in self.records)
        total = len(self.records)
        average = sum(item.risk_score for item in self.records) / total if total else 0.0
        high_risk = counts[Decision.HIGH_RISK.value]
        return SimpleNamespace(
            total_applications=total,
            approved_applications=counts[Decision.APPROVE.value],
            manual_reviews=counts[Decision.MANUAL_REVIEW.value],
            high_risk_applications=high_risk,
            average_risk_score=average,
            fraud_high_risk_rate=high_risk / total if total else 0.0,
            decision_counts={decision.value: counts[decision.value] for decision in Decision},
        )

    def ping(self) -> bool:
        return self.available

    def record_review(
        self, application_id: str, outcome: InvestigatorOutcome, reviewer_id: str
    ) -> SimpleNamespace | None:
        del reviewer_id
        if self.get_application(application_id) is None:
            return None
        self.reviews[application_id] = outcome
        return SimpleNamespace(
            application_id=application_id,
            outcome=outcome,
            reviewed_at=datetime.now(UTC),
        )

    def learning_status(self) -> SimpleNamespace:
        counts = Counter(outcome.value for outcome in self.reviews.values())
        return SimpleNamespace(
            reviewed_applications=len(self.reviews),
            confirmed_fraud=counts[InvestigatorOutcome.CONFIRMED_FRAUD.value],
            legitimate=counts[InvestigatorOutcome.LEGITIMATE.value],
            inconclusive=counts[InvestigatorOutcome.INCONCLUSIVE.value],
            false_positive_reviews=0,
            missed_fraud_reviews=0,
            reviewed_alerts=0,
            false_positive_review_rate=0.0,
            minimum_feedback_required=100,
            retraining_recommended=False,
            governance_status="COLLECTING_FEEDBACK",
        )

    def reset_demo_namespace(self) -> int:
        count = len(self.records)
        self.records.clear()
        self.reviews.clear()
        return count


def event_payload(application_id: str = "APP-001", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "application_id": application_id,
        "user_id": "USER-001",
        "event_timestamp": "2026-08-20T10:00:00Z",
        "requested_loan_amount": 5_000.0,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def api() -> tuple[TestClient, FakeRepository]:
    repository = FakeRepository()
    supervised = SupervisedModel(
        estimator=RatioEstimator(),
        threshold=0.42,
        model_name="ratio-test",
        model_version="api-test-v1",
        manifest={},
    )
    risk_config, version = load_risk_config("config/risk.yaml")
    decisioning = DecisioningService(
        state_store=MemoryRealtimeStateStore(),
        supervised=supervised,
        anomaly=FixedAnomaly(),  # type: ignore[arg-type]
        risk_engine=RiskEngine(risk_config, version),
        sink=repository,
    )
    settings = Settings(
        app_env="test",
        pseudonym_key="test-key-that-is-long-enough-for-validation",
        cors_origins=("https://dashboard.test",),
    )
    app = create_app(
        settings=settings,
        container=AppContainer(
            settings=settings,
            decisioning=decisioning,
            models=FakeRegistry(),
            repository=repository,
            simulator=DemoSimulator(decisioning, resetter=repository),
        ),
    )
    with TestClient(app) as client:
        yield client, repository


def test_health_distinguishes_liveness_and_readiness(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client, repository = api
    healthy = client.get("/health")
    assert healthy.status_code == 200
    assert healthy.json() == {
        "status": "ok",
        "liveness": {"status": "alive"},
        "readiness": {
            "status": "ready",
            "checks": {
                "models": True,
                "persistence": True,
                "state": True,
                "startup": True,
            },
        },
    }

    repository.available = False
    degraded = client.get("/health")
    assert degraded.status_code == 200
    assert degraded.json()["liveness"]["status"] == "alive"
    assert degraded.json()["readiness"]["status"] == "not_ready"
    assert degraded.json()["readiness"]["checks"]["persistence"] is False

    class UnavailableState:
        def ping(self) -> bool:
            return False

    decisioning = client.app.state.container.decisioning
    assert decisioning is not None
    decisioning.state_store = UnavailableState()
    state_degraded = client.get("/health")
    assert state_degraded.json()["readiness"]["checks"]["state"] is False


def test_model_info_is_safe_and_complete(api: tuple[TestClient, FakeRepository]) -> None:
    response = api[0].get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_version"] == "api-test-v1"
    assert response.json()["metrics"]["recall"] == 0.9
    assert response.json()["prototype_only"] is True


def test_predict_returns_complete_assessment_and_persists(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client, repository = api
    response = client.post(
        "/predict",
        json=event_payload(
            income=80_000.0,
            account_age_days=900,
            device_id="DEVICE-TRUSTED",
            ip_address="203.0.113.8",
        ),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["application_id"] == "APP-001"
    assert set(body["component_scores"]) == {
        "supervised_probability",
        "anomaly_score",
        "behavioral_risk",
        "graph_risk",
    }
    assert body["model_version"] == "api-test-v1"
    assert body["risk_config_version"].startswith("risk-v")
    assert body["feature_schema_version"]
    assert len(repository.records) == 1
    assert repository.records[0].application_id == "APP-001"


def test_new_user_and_missing_optional_fields_are_scoreable(
    api: tuple[TestClient, FakeRepository],
) -> None:
    response = api[0].post("/predict", json=event_payload("APP-NEW", user_id="NEW-USER"))
    assert response.status_code == 200
    assert response.json()["application_id"] == "APP-NEW"


def test_extreme_bounded_value_scores_but_out_of_range_is_sanitized(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client = api[0]
    bounded = client.post(
        "/predict",
        json=event_payload("APP-EXTREME", requested_loan_amount=10_000_000.0, income=1.0),
    )
    assert bounded.status_code == 200
    rejected = client.post(
        "/predict",
        json=event_payload("APP-TOO-LARGE", requested_loan_amount=10_000_001.0),
    )
    assert rejected.status_code == 422
    serialized = rejected.text
    assert "10000001" not in serialized
    assert "input" not in serialized
    assert rejected.headers["x-request-id"]


@pytest.mark.parametrize(
    "payload",
    [
        {"application_id": "APP-BAD"},
        event_payload(ip_address="999.1.2.3"),
        event_payload(unexpected_secret="must-not-be-reflected"),
        event_payload(event_timestamp="2026-08-20T10:00:00"),
        event_payload(requested_loan_amount="5000"),
    ],
)
def test_malformed_requests_are_rejected_without_echoing_values(
    api: tuple[TestClient, FakeRepository], payload: dict[str, object]
) -> None:
    response = api[0].post("/predict", json=payload)
    assert response.status_code == 422
    assert "must-not-be-reflected" not in response.text
    for error in response.json()["detail"]:
        assert set(error) == {"type", "loc", "msg"}


def test_oversized_request_is_rejected_before_validation(
    api: tuple[TestClient, FakeRepository],
) -> None:
    response = api[0].post(
        "/predict",
        content=b"{" + b"x" * (64 * 1024) + b"}",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large"
    assert response.headers["x-request-id"]


def test_repeated_applications_and_shared_device_affect_real_time_assessment(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client = api[0]
    started = datetime(2026, 8, 20, 10, tzinfo=UTC)

    repeated_scores = []
    for index in range(4):
        response = client.post(
            "/predict",
            json=event_payload(
                f"APP-REPEAT-{index}",
                user_id="USER-REPEAT",
                device_id="DEVICE-REPEAT",
                event_timestamp=(started + timedelta(minutes=index)).isoformat(),
                income=40_000.0,
            ),
        )
        assert response.status_code == 200
        repeated_scores.append(response.json()["risk_score"])
    assert repeated_scores[-1] > repeated_scores[0]

    shared_scores = []
    for index in range(3):
        response = client.post(
            "/predict",
            json=event_payload(
                f"APP-SHARED-{index}",
                user_id=f"USER-SHARED-{index}",
                device_id="DEVICE-SHARED",
                event_timestamp=(started + timedelta(hours=1, minutes=index)).isoformat(),
                income=40_000.0,
            ),
        )
        assert response.status_code == 200
        shared_scores.append(response.json()["risk_score"])
    assert shared_scores[-1] > shared_scores[0]


def test_dashboard_list_detail_filter_and_analytics(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client = api[0]
    assert (
        client.post(
            "/predict",
            json=event_payload("APP-DASH-1", income=80_000.0, channel="MOBILE"),
        ).status_code
        == 200
    )
    assert (
        client.post("/predict", json=event_payload("APP-DASH-2", income=200.0)).status_code == 200
    )

    listing = client.get("/applications?limit=1&offset=0")
    assert listing.status_code == 200
    assert len(listing.json()) == 1
    assert listing.json()[0]["application_id"] == "APP-DASH-2"
    assert "user_id" not in listing.text

    detail = client.get("/applications/APP-DASH-1")
    assert detail.status_code == 200
    assert detail.json()["application_id"] == "APP-DASH-1"
    assert detail.json()["channel"] == "MOBILE"
    expected_actions = {
        "APPROVE": "CONTINUE_WITH_STANDARD_CHECKS",
        "MANUAL_REVIEW": "REQUIRE_ENHANCED_VERIFICATION",
        "HIGH_RISK": "HOLD_AND_INVESTIGATE",
    }
    assert detail.json()["recommended_action"] == expected_actions[detail.json()["decision"]]
    assert "reasons" in detail.json()
    assert client.get("/applications/UNKNOWN").status_code == 404

    decision = client.get("/applications").json()[0]["decision"]
    filtered = client.get(f"/applications?decision={decision}")
    assert filtered.status_code == 200
    assert all(item["decision"] == decision for item in filtered.json())

    analytics = client.get("/analytics")
    assert analytics.status_code == 200
    assert analytics.json()["total_applications"] == 2
    assert sum(analytics.json()["decision_counts"].values()) == 2


def test_investigator_feedback_updates_governed_learning_status(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client = api[0]
    assert client.post("/predict", json=event_payload("APP-REVIEW-1")).status_code == 200

    review = client.post(
        "/applications/APP-REVIEW-1/review",
        json={"outcome": "CONFIRMED_FRAUD", "reviewer_id": "REVIEWER-DEMO"},
    )
    assert review.status_code == 200
    assert review.json()["outcome"] == "CONFIRMED_FRAUD"
    assert "reviewer" not in review.text.lower()

    learning = client.get("/learning/status")
    assert learning.status_code == 200
    assert learning.json()["reviewed_applications"] == 1
    assert learning.json()["confirmed_fraud"] == 1
    assert learning.json()["governance_status"] == "COLLECTING_FEEDBACK"

    assert client.post(
        "/applications/UNKNOWN/review",
        json={"outcome": "LEGITIMATE", "reviewer_id": "REVIEWER-DEMO"},
    ).status_code == 404


def test_cors_uses_configured_origins(api: tuple[TestClient, FakeRepository]) -> None:
    allowed = api[0].options(
        "/predict",
        headers={
            "Origin": "https://dashboard.test",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://dashboard.test"

    denied = api[0].options(
        "/predict",
        headers={"Origin": "https://evil.test", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in denied.headers


def test_demo_run_status_stop_and_reset_use_real_decisioning_path(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client, repository = api
    started = client.post(
        "/demo/run",
        json={"scenario": "normal", "interval_ms": 50, "repeat": 1},
    )
    assert started.status_code == 200
    assert started.json()["total"] == 1

    status_response = client.get("/demo/status")
    assert status_response.status_code == 200
    assert status_response.json()["processed"] == 1
    assert repository.records[0].application_id == "APP-NORMAL-001"

    assert client.post("/demo/stop").status_code == 200
    reset = client.post("/demo/reset")
    assert reset.status_code == 200
    assert reset.json()["processed"] == 0
    assert repository.records == []


def test_demo_request_validation_and_running_conflict_are_safe(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client = api[0]
    invalid = client.post("/demo/run", json={"scenario": "unknown", "interval_ms": 1})
    assert invalid.status_code == 422
    assert "unknown" not in invalid.text

    first = client.post(
        "/demo/run",
        json={"scenario": "fraud_ring", "interval_ms": 500, "repeat": 2},
    )
    assert first.status_code == 200
    second = client.post("/demo/run", json={"scenario": "normal"})
    assert second.status_code == 409
    client.post("/demo/stop")


def test_random_demo_contract_starts_bounded_seeded_stream(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client, _ = api
    response = client.post(
        "/demo/random/run",
        json={
            "count": 12,
            "interval_ms": 50,
            "seed": 77,
            "normal_percent": 50,
            "suspicious_percent": 25,
            "fraud_percent": 25,
        },
    )
    assert response.status_code == 200
    assert response.json()["scenario"] == "random"
    assert response.json()["total"] == 12
    client.post("/demo/stop")

    invalid = client.post(
        "/demo/random/run",
        json={"normal_percent": 80, "suspicious_percent": 15, "fraud_percent": 10},
    )
    assert invalid.status_code == 422


def test_dependency_override_and_internal_errors_are_sanitized(
    api: tuple[TestClient, FakeRepository],
) -> None:
    client = api[0]

    class ExplodingDecisioning:
        def assess(self, _event: LoanApplicationEvent) -> FraudAssessment:
            raise RuntimeError("database-password=must-never-leak")

    client.app.dependency_overrides[get_decisioning] = lambda: ExplodingDecisioning()
    with TestClient(client.app, raise_server_exceptions=False) as sanitized_client:
        response = sanitized_client.post("/predict", json=event_payload("APP-ERROR"))
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert "must-never-leak" not in response.text
