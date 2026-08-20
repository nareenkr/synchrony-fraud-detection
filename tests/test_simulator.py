from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.services.simulator import DemoSimulator


class FakeState:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class FakeDecisioning:
    def __init__(self) -> None:
        self.state_store = FakeState()
        self.events: list[Any] = []

    def assess(self, event: Any, *, now: Any = None) -> Any:
        self.events.append(event)
        return FakeAssessment(event.application_id)


@dataclass
class FakeAssessment:
    application_id: str


class FakeResetter:
    def __init__(self) -> None:
        self.calls = 0

    def reset_demo_namespace(self) -> int:
        self.calls += 1
        return 0


def test_mixed_simulation_processes_all_scenarios_in_order() -> None:
    decisioning = FakeDecisioning()
    observed: list[str] = []
    simulator = DemoSimulator(
        decisioning,  # type: ignore[arg-type]
        on_assessment=lambda result: observed.append(result.application_id),
    )

    started = simulator.start("mixed", interval_ms=50)
    finished = simulator.wait(timeout=3)

    assert started.total >= 10
    assert finished.processed == finished.total
    assert finished.error is None
    assert observed[0] == "APP-NORMAL-001"
    assert any("SUSPICIOUS" in value for value in observed)
    assert any("RING" in value for value in observed)


def test_repeated_run_has_unique_deterministic_ids() -> None:
    events = DemoSimulator._events("normal", repeat=2)
    assert [event.application_id for event in events] == [
        "APP-NORMAL-001",
        "APP-NORMAL-001-R2",
    ]
    assert events[1].event_timestamp > events[0].event_timestamp


def test_stop_reset_and_validation() -> None:
    decisioning = FakeDecisioning()
    resetter = FakeResetter()
    simulator = DemoSimulator(decisioning, resetter=resetter)  # type: ignore[arg-type]

    simulator.start("fraud_ring", interval_ms=100)
    simulator.stop()
    status = simulator.reset()
    assert status.running is False
    assert status.processed == 0
    assert decisioning.state_store.reset_count == 1
    assert resetter.calls == 1

    for kwargs in (
        {"scenario": "unknown"},
        {"interval_ms": 1},
        {"repeat": 0},
    ):
        try:
            simulator.start(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected validation error for {kwargs}")


def test_random_stream_is_seeded_bounded_and_preserves_profile_correlations() -> None:
    first = DemoSimulator._random_events(
        count=100,
        interval_ms=250,
        seed=20260820,
        normal_percent=80,
        suspicious_percent=15,
    )
    second = DemoSimulator._random_events(
        count=100,
        interval_ms=250,
        seed=20260820,
        normal_percent=80,
        suspicious_percent=15,
    )

    assert first == second
    assert len(first) == 100
    assert len({event.application_id for event in first}) == 100
    assert sum("-N-" in event.application_id for event in first) == 80
    assert sum("-S-" in event.application_id for event in first) == 15
    assert sum("-F-" in event.application_id for event in first) == 5
    suspicious = [event for event in first if "-S-" in event.application_id]
    assert len({event.device_id for event in suspicious}) < len(suspicious)
    assert all(
        later.event_timestamp > earlier.event_timestamp
        for earlier, later in zip(first, first[1:], strict=False)
    )


def test_random_stream_runs_through_decisioning_and_validates_mix() -> None:
    decisioning = FakeDecisioning()
    simulator = DemoSimulator(decisioning)  # type: ignore[arg-type]

    started = simulator.start_random(
        count=12,
        interval_ms=50,
        seed=7,
        normal_percent=50,
        suspicious_percent=25,
        fraud_percent=25,
    )
    finished = simulator.wait(timeout=2)
    assert started.scenario == "random"
    assert finished.processed == 12
    assert len(decisioning.events) == 12

    try:
        simulator.start_random(normal_percent=80, suspicious_percent=15, fraud_percent=10)
    except ValueError as exc:
        assert "total 100" in str(exc)
    else:
        raise AssertionError("invalid random profile mix was accepted")
