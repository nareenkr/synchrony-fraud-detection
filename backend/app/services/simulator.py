"""Deterministic background replay for the live fraud-monitoring demo."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread
from typing import Protocol

from backend.app.schemas import FraudAssessment, LendingChannel, LoanApplicationEvent
from training.prepare_data import demo_scenarios

from .decisioning import DecisioningService


class DemoResetter(Protocol):
    def reset_demo_namespace(self) -> int: ...


@dataclass(frozen=True, slots=True)
class SimulatorStatus:
    running: bool
    scenario: str | None
    processed: int
    total: int
    last_application_id: str | None
    error: str | None


class DemoSimulator:
    """Replay canonical scenarios through the real decisioning path over time."""

    _VALID_SCENARIOS = frozenset({"normal", "suspicious", "fraud_ring", "mixed"})

    def __init__(
        self,
        decisioning: DecisioningService,
        *,
        resetter: DemoResetter | None = None,
        on_assessment: Callable[[FraudAssessment], None] | None = None,
    ) -> None:
        self._decisioning = decisioning
        self._resetter = resetter
        self._on_assessment = on_assessment
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._scenario: str | None = None
        self._processed = 0
        self._total = 0
        self._last_application_id: str | None = None
        self._error: str | None = None

    def start(
        self,
        scenario: str = "mixed",
        *,
        interval_ms: int = 750,
        repeat: int = 1,
    ) -> SimulatorStatus:
        if scenario not in self._VALID_SCENARIOS:
            raise ValueError(f"scenario must be one of {sorted(self._VALID_SCENARIOS)}")
        if not 50 <= interval_ms <= 60_000:
            raise ValueError("interval_ms must be between 50 and 60000")
        if not 1 <= repeat <= 10:
            raise ValueError("repeat must be between 1 and 10")
        events = self._events(scenario, repeat)
        return self._start_events(events, interval_ms=interval_ms, scenario=scenario)

    def start_random(
        self,
        *,
        count: int = 100,
        interval_ms: int = 500,
        seed: int = 20_260_820,
        normal_percent: int = 80,
        suspicious_percent: int = 15,
        fraud_percent: int = 5,
    ) -> SimulatorStatus:
        """Generate a bounded, reproducible stream of correlated synthetic events."""

        if not 1 <= count <= 5_000:
            raise ValueError("count must be between 1 and 5000")
        if not 50 <= interval_ms <= 60_000:
            raise ValueError("interval_ms must be between 50 and 60000")
        percentages = (normal_percent, suspicious_percent, fraud_percent)
        if any(value < 0 or value > 100 for value in percentages) or sum(percentages) != 100:
            raise ValueError("profile percentages must be between 0 and 100 and total 100")
        events = self._random_events(
            count=count,
            interval_ms=interval_ms,
            seed=seed,
            normal_percent=normal_percent,
            suspicious_percent=suspicious_percent,
        )
        return self._start_events(events, interval_ms=interval_ms, scenario="random")

    def _start_events(
        self,
        events: tuple[LoanApplicationEvent, ...],
        *,
        interval_ms: int,
        scenario: str,
    ) -> SimulatorStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a demo simulation is already running")
            self._stop.clear()
            self._scenario = scenario
            self._processed = 0
            self._total = len(events)
            self._last_application_id = None
            self._error = None
            self._thread = Thread(
                target=self._run,
                args=(events, interval_ms / 1000.0),
                name="synchrony-demo-simulator",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self) -> SimulatorStatus:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        return self.status()

    def reset(self) -> SimulatorStatus:
        self.stop()
        self._decisioning.state_store.reset()
        if self._resetter is not None:
            self._resetter.reset_demo_namespace()
        with self._lock:
            self._scenario = None
            self._processed = 0
            self._total = 0
            self._last_application_id = None
            self._error = None
        return self.status()

    def wait(self, timeout: float | None = None) -> SimulatorStatus:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        return self.status()

    def status(self) -> SimulatorStatus:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return SimulatorStatus(
                running=running,
                scenario=self._scenario,
                processed=self._processed,
                total=self._total,
                last_application_id=self._last_application_id,
                error=self._error,
            )

    def _run(self, events: tuple[LoanApplicationEvent, ...], interval_seconds: float) -> None:
        try:
            for index, event in enumerate(events):
                if self._stop.is_set():
                    break
                assessment = self._decisioning.assess(event, now=event.event_timestamp)
                if self._on_assessment is not None:
                    self._on_assessment(assessment)
                with self._lock:
                    self._processed = index + 1
                    self._last_application_id = event.application_id
                if index + 1 < len(events) and self._stop.wait(interval_seconds):
                    break
        except Exception as exc:  # surfaced through status without leaking a request body
            with self._lock:
                self._error = f"{type(exc).__name__}: simulation stopped"

    @staticmethod
    def _events(scenario: str, repeat: int) -> tuple[LoanApplicationEvent, ...]:
        scenarios = demo_scenarios()
        base = (
            scenarios["normal"] + scenarios["suspicious"] + scenarios["fraud_ring"]
            if scenario == "mixed"
            else scenarios[scenario]
        )
        repeated: list[LoanApplicationEvent] = []
        for repetition in range(repeat):
            for event in base:
                if repetition == 0:
                    repeated.append(event)
                else:
                    repeated.append(
                        event.model_copy(
                            update={
                                "application_id": f"{event.application_id}-R{repetition + 1}",
                                "event_timestamp": event.event_timestamp
                                + timedelta(days=repetition),
                            }
                        )
                    )
        return tuple(repeated)

    @staticmethod
    def _random_events(
        *,
        count: int,
        interval_ms: int,
        seed: int,
        normal_percent: int,
        suspicious_percent: int,
    ) -> tuple[LoanApplicationEvent, ...]:
        """Build random-looking events while preserving behavioral/graph correlations."""

        rng = random.Random(seed)
        normal_count = count * normal_percent // 100
        suspicious_count = count * suspicious_percent // 100
        profiles = (
            ["normal"] * normal_count
            + ["suspicious"] * suspicious_count
            + ["fraud_ring"] * (count - normal_count - suspicious_count)
        )
        rng.shuffle(profiles)
        profile_indexes = {"normal": 0, "suspicious": 0, "fraud_ring": 0}
        base_time = datetime(2026, 8, 20, 12, tzinfo=UTC)
        events: list[LoanApplicationEvent] = []

        for sequence, profile in enumerate(profiles, start=1):
            profile_index = profile_indexes[profile]
            profile_indexes[profile] += 1
            event_time = base_time + timedelta(milliseconds=interval_ms * (sequence - 1))
            suffix = f"{seed}-{sequence:05d}"
            channel = rng.choice(tuple(LendingChannel))

            if profile == "normal":
                income = rng.uniform(45_000, 180_000)
                requested = rng.uniform(1_000, min(25_000, income * 0.22))
                transaction = rng.uniform(50, 2_500)
                balance = rng.uniform(max(transaction, 3_000), 120_000)
                event = LoanApplicationEvent(
                    application_id=f"APP-RANDOM-N-{suffix}",
                    user_id=f"USR-RANDOM-N-{seed}-{profile_index:05d}",
                    event_timestamp=event_time,
                    requested_loan_amount=round(requested, 2),
                    channel=channel,
                    income=round(income, 2),
                    debt_to_income_ratio=round(rng.uniform(0.05, 0.35), 3),
                    account_age_days=rng.randint(365, 3_650),
                    bank_account_age_days=rng.randint(500, 4_000),
                    device_id=f"DEV-RANDOM-N-{seed}-{profile_index:05d}",
                    ip_address=f"198.51.100.{10 + profile_index % 240}",
                    bank_account_id=f"BANK-RANDOM-N-{seed}-{profile_index:05d}",
                    geographic_region=rng.choice(("NORTH", "SOUTH", "EAST", "WEST")),
                    device_changes_30d=rng.randint(0, 1),
                    login_frequency_24h=rng.randint(1, 8),
                    failed_login_attempts_24h=rng.randint(0, 1),
                    previous_rejected_applications=0,
                    unusual_login_location=False,
                    transaction_amount=round(transaction, 2),
                    transaction_frequency_24h=rng.randint(1, 12),
                    transaction_amount_deviation=round(rng.uniform(0.0, 0.5), 3),
                    origin_balance_before=round(balance, 2),
                    origin_balance_after=round(balance - transaction, 2),
                )
            elif profile == "suspicious":
                cluster = profile_index // 4
                income = rng.uniform(28_000, 85_000)
                event = LoanApplicationEvent(
                    application_id=f"APP-RANDOM-S-{suffix}",
                    user_id=f"USR-RANDOM-S-{seed}-{cluster:04d}",
                    event_timestamp=event_time,
                    requested_loan_amount=round(income * rng.uniform(0.72, 1.35), 2),
                    channel=channel,
                    income=round(income, 2),
                    debt_to_income_ratio=round(rng.uniform(0.65, 1.25), 3),
                    account_age_days=rng.randint(0, 30),
                    bank_account_age_days=rng.randint(0, 45),
                    device_id=f"DEV-RANDOM-S-{seed}-{cluster:04d}",
                    ip_address=f"203.0.113.{20 + cluster % 220}",
                    bank_account_id=f"BANK-RANDOM-S-{seed}-{cluster:04d}",
                    geographic_region=rng.choice(("EAST", "NORTH")),
                    device_changes_30d=rng.randint(3, 8),
                    login_frequency_24h=rng.randint(18, 55),
                    failed_login_attempts_24h=rng.randint(3, 10),
                    previous_rejected_applications=rng.randint(1, 4),
                    unusual_login_location=True,
                    transaction_amount=round(income * rng.uniform(0.25, 0.8), 2),
                    transaction_frequency_24h=rng.randint(20, 70),
                    transaction_amount_deviation=round(rng.uniform(2.5, 7.0), 3),
                )
            else:
                ring = profile_index // 6
                income = rng.uniform(20_000, 60_000)
                shared_bank = rng.random() < 0.45
                event = LoanApplicationEvent(
                    application_id=f"APP-RANDOM-F-{suffix}",
                    user_id=f"USR-RANDOM-F-{seed}-{profile_index:05d}",
                    event_timestamp=event_time,
                    requested_loan_amount=round(income * rng.uniform(1.8, 3.5), 2),
                    channel=channel,
                    income=round(income, 2),
                    debt_to_income_ratio=round(rng.uniform(1.5, 4.0), 3),
                    account_age_days=rng.randint(0, 7),
                    bank_account_age_days=rng.randint(0, 10),
                    device_id=f"DEV-RANDOM-RING-{seed}-{ring:04d}",
                    ip_address=f"192.0.2.{30 + ring % 210}",
                    bank_account_id=(
                        f"BANK-RANDOM-RING-{seed}-{ring:04d}"
                        if shared_bank
                        else f"BANK-RANDOM-F-{seed}-{profile_index:05d}"
                    ),
                    geographic_region="NORTH",
                    device_changes_30d=rng.randint(7, 15),
                    login_frequency_24h=rng.randint(60, 140),
                    failed_login_attempts_24h=rng.randint(8, 25),
                    previous_rejected_applications=rng.randint(3, 8),
                    unusual_login_location=True,
                    transaction_amount=round(income * rng.uniform(1.0, 2.2), 2),
                    transaction_frequency_24h=rng.randint(80, 180),
                    transaction_amount_deviation=round(rng.uniform(8.0, 18.0), 3),
                )
            events.append(event)
        return tuple(events)
