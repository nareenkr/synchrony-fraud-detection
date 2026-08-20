"""End-to-end regression for the retained artifacts and canonical demo contract."""

from backend.app.fraud.risk_engine import RiskEngine, load_risk_config
from backend.app.schemas import Decision
from backend.app.services.decisioning import DecisioningService
from backend.app.services.model_registry import ModelRegistry
from backend.app.state import MemoryRealtimeStateStore
from training.prepare_data import demo_scenarios


def _run_mixed_demo() -> list[tuple[str, float, Decision]]:
    registry = ModelRegistry.load("artifacts/supervised-v1", "artifacts/anomaly-v1.joblib")
    risk_config, version = load_risk_config("config/risk.yaml")
    service = DecisioningService(
        state_store=MemoryRealtimeStateStore(),
        supervised=registry.supervised,
        anomaly=registry.anomaly,
        risk_engine=RiskEngine(risk_config, version),
    )
    scenarios = demo_scenarios()
    events = scenarios["normal"] + scenarios["suspicious"] + scenarios["fraud_ring"]
    return [
        (
            event.application_id,
            assessment.risk_score,
            assessment.decision,
        )
        for event in events
        for assessment in (service.assess(event, now=event.event_timestamp),)
    ]


def test_retained_artifacts_preserve_demo_bands_and_determinism() -> None:
    first = _run_mixed_demo()
    second = _run_mixed_demo()

    assert first == second
    assert len(first) == 11
    assert first[0][1] < 40
    assert first[0][2] is Decision.APPROVE
    assert all(
        40 <= score < 70 and decision is Decision.MANUAL_REVIEW
        for _, score, decision in first[1:5]
    )
    assert all(score >= 70 and decision is Decision.HIGH_RISK for _, score, decision in first[5:])
