"""Behavioral state contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Barrier, Thread

import pytest

from backend.app.state import MemoryRealtimeStateStore, RealtimeStateStore


@dataclass
class MutableClock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value


def event(user: str, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "user_id": user,
        "device_id": f"device-{user}",
        "ip_address": f"10.0.0.{len(user)}",
        "bank_account_id": f"bank-{user}",
    }
    value.update(overrides)
    return value


def test_snapshot_is_prior_only_then_records_current_event() -> None:
    now = datetime(2026, 8, 19, 12, tzinfo=UTC)
    store = MemoryRealtimeStateStore(clock=lambda: now)
    current = event("u1")

    first = store.snapshot_and_record(current)
    second = store.snapshot_and_record(current)

    assert first.applications_last_hour == 0
    assert first.applications_last_day == 0
    assert second.applications_last_hour == 1
    assert second.applications_last_24h == 1
    assert len(store) == 2


def test_velocity_windows_include_boundaries_and_expire() -> None:
    clock = MutableClock(datetime(2026, 8, 19, 12, tzinfo=UTC))
    store = MemoryRealtimeStateStore(clock=clock)
    store.record_event(event("u1"), now=clock.value - timedelta(hours=1))
    store.record_event(event("u1"), now=clock.value - timedelta(hours=2))

    snapshot = store.get_snapshot(event("u1"))
    assert snapshot.applications_last_hour == 1
    assert snapshot.applications_last_day == 2

    clock.value += timedelta(days=1, seconds=1)
    expired = store.get_snapshot(event("u1"))
    assert expired.applications_last_hour == 0
    assert expired.applications_last_day == 0


def test_distinct_entity_users_are_based_only_on_prior_events() -> None:
    now = datetime(2026, 8, 19, 12, tzinfo=UTC)
    store = MemoryRealtimeStateStore(clock=lambda: now)
    shared = {"device_id": "d-shared", "ip_address": "1.2.3.4", "bank_account_id": "b-shared"}
    store.snapshot_and_record(event("u1", **shared))
    store.snapshot_and_record(event("u1", **shared))

    before_u2 = store.snapshot_and_record(event("u2", **shared))
    after_u2 = store.get_snapshot(event("u3", **shared))

    assert before_u2.distinct_users_for_device == 1
    assert before_u2.distinct_users_for_ip == 1
    assert before_u2.distinct_users_for_bank == 1
    assert after_u2.shared_device_user_count == 2
    assert after_u2.shared_ip_user_count == 2
    assert after_u2.shared_bank_user_count == 2


def test_device_change_is_detected_and_then_counted_as_history() -> None:
    now = datetime(2026, 8, 19, 12, tzinfo=UTC)
    store = MemoryRealtimeStateStore(clock=lambda: now)
    store.snapshot_and_record(event("u1", device_id="old"))

    changed = store.snapshot_and_record(event("u1", device_id="new"))
    after = store.get_snapshot(event("u1", device_id="new"))

    assert changed.last_device_id == "old"
    assert changed.device_changed is True
    assert changed.device_changes_30d == 0
    assert after.device_changed is False
    assert after.device_changes_30d == 1


def test_login_and_failed_login_tracking() -> None:
    now = datetime(2026, 8, 19, 12, tzinfo=UTC)
    store = MemoryRealtimeStateStore(clock=lambda: now)
    store.record_event(event("u1", event_type="login"))
    store.record_event(event("u1", event_type="failed-login", failed_login_count=3))
    store.record_event(event("u2", event_type="failed_login"))

    snapshot = store.get_snapshot(event("u1"))
    assert snapshot.login_frequency_24h == 2
    assert snapshot.failed_login_attempts_24h == 3
    assert snapshot.applications_last_day == 0


def test_ttl_pruning_and_reset() -> None:
    clock = MutableClock(datetime(2026, 7, 1, tzinfo=UTC))
    store = MemoryRealtimeStateStore(clock=clock)
    store.record_event(event("old"))
    clock.value += timedelta(days=30, seconds=1)

    assert store.prune() == 1
    assert len(store) == 0
    store.record_event(event("new"))
    store.reset()
    assert len(store) == 0


def test_future_events_are_invisible_to_point_in_time_snapshot() -> None:
    now = datetime(2026, 8, 19, 12, tzinfo=UTC)
    store = MemoryRealtimeStateStore(clock=lambda: now)
    store.record_event(event("u1"), now=now + timedelta(hours=1))

    assert store.get_snapshot(event("u1"), now=now).applications_last_day == 0


def test_mapping_iso_timestamp_and_object_inputs_are_supported() -> None:
    @dataclass
    class CanonicalLikeEvent:
        applicant_id: str
        device_fingerprint: str
        event_timestamp: datetime

    now = datetime(2026, 8, 19, 12, tzinfo=UTC)
    store = MemoryRealtimeStateStore()
    store.snapshot_and_record({"customer_id": "u1", "timestamp": "2026-08-19T12:00:00Z"})
    snapshot = store.snapshot_and_record(CanonicalLikeEvent("u1", "device", now), now=now)
    assert snapshot.applications_last_hour == 1
    assert isinstance(store, RealtimeStateStore)


def test_snapshot_and_record_is_atomic_across_threads() -> None:
    now = datetime(2026, 8, 19, 12, tzinfo=UTC)
    store = MemoryRealtimeStateStore(clock=lambda: now)
    barrier = Barrier(8)
    observed: list[int] = []

    def worker() -> None:
        barrier.wait()
        observed.append(store.snapshot_and_record(event("u1")).applications_last_hour)

    threads = [Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(observed) == list(range(8))


def test_invalid_retention_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must cover"):
        MemoryRealtimeStateStore(ttl=timedelta(hours=1))
