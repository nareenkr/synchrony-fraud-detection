"""Thread-safe, bounded in-process implementation of real-time state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Any

from .base import StateSnapshot

_MISSING = object()


@dataclass(frozen=True, slots=True)
class _RecordedEvent:
    occurred_at: datetime
    user_id: str | None
    device_id: str | None
    ip_address: str | None
    bank_account_id: str | None
    event_kind: str
    failed_login_count: int
    device_changed: bool


class MemoryRealtimeStateStore:
    """A local single-process state store with point-in-time semantics.

    Events are retained for ``ttl`` and are filtered by their occurrence time,
    not insertion order.  This supports deterministic chronological replay as
    well as normal online requests.  An explicit ``now`` takes precedence over
    an event timestamp; otherwise the event timestamp is used when present and
    the injectable clock is the final fallback.

    The adapter is thread-safe within one process.  It deliberately makes no
    cross-worker guarantee; use an implementation of ``RealtimeStateStore``
    backed by Redis before deploying multiple workers.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        ttl: timedelta = timedelta(days=30),
        application_hour_window: timedelta = timedelta(hours=1),
        application_day_window: timedelta = timedelta(days=1),
        login_window: timedelta = timedelta(days=1),
        device_change_window: timedelta = timedelta(days=30),
    ) -> None:
        windows = {
            "ttl": ttl,
            "application_hour_window": application_hour_window,
            "application_day_window": application_day_window,
            "login_window": login_window,
            "device_change_window": device_change_window,
        }
        if any(value <= timedelta(0) for value in windows.values()):
            raise ValueError("state TTL and windows must be positive")
        if ttl < max(value for name, value in windows.items() if name != "ttl"):
            raise ValueError("state TTL must cover every configured feature window")

        self._clock = clock or (lambda: datetime.now(UTC))
        self._ttl = ttl
        self._application_hour_window = application_hour_window
        self._application_day_window = application_day_window
        self._login_window = login_window
        self._device_change_window = device_change_window
        self._events: list[_RecordedEvent] = []
        self._lock = RLock()

    def get_snapshot(self, event: Any, *, now: datetime | None = None) -> StateSnapshot:
        as_of = self._resolve_time(event, now)
        with self._lock:
            self._prune_locked(as_of)
            return self._snapshot_locked(event, as_of)

    # Short aliases retain the language used in the architecture document.
    def snapshot(self, event: Any, *, now: datetime | None = None) -> StateSnapshot:
        return self.get_snapshot(event, now=now)

    def record_event(self, event: Any, *, now: datetime | None = None) -> None:
        occurred_at = self._resolve_time(event, now)
        with self._lock:
            self._prune_locked(occurred_at)
            self._record_locked(event, occurred_at)

    def record(self, event: Any, *, now: datetime | None = None) -> None:
        self.record_event(event, now=now)

    def snapshot_and_record(self, event: Any, *, now: datetime | None = None) -> StateSnapshot:
        occurred_at = self._resolve_time(event, now)
        with self._lock:
            self._prune_locked(occurred_at)
            snapshot = self._snapshot_locked(event, occurred_at)
            self._record_locked(event, occurred_at)
            return snapshot

    def prune(self, *, now: datetime | None = None) -> int:
        as_of = self._normalise_datetime(now if now is not None else self._clock())
        with self._lock:
            return self._prune_locked(as_of)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def _snapshot_locked(self, event: Any, as_of: datetime) -> StateSnapshot:
        user_id = self._identifier(event, "user_id", "customer_id", "applicant_id")
        device_id = self._identifier(event, "device_id", "device_fingerprint")
        ip_address = self._identifier(event, "ip_address", "ip")
        bank_id = self._identifier(event, "bank_account_id", "bank_id", "bank_account")

        visible = [record for record in self._events if record.occurred_at <= as_of]
        user_records = (
            [record for record in visible if record.user_id == user_id] if user_id else []
        )

        def since(record: _RecordedEvent, window: timedelta) -> bool:
            return record.occurred_at >= as_of - window

        applications_last_hour = sum(
            record.event_kind == "application" and since(record, self._application_hour_window)
            for record in user_records
        )
        applications_last_day = sum(
            record.event_kind == "application" and since(record, self._application_day_window)
            for record in user_records
        )
        login_frequency = sum(
            record.event_kind in {"login", "failed_login"} and since(record, self._login_window)
            for record in user_records
        )
        failed_logins = sum(
            record.failed_login_count
            for record in user_records
            if since(record, self._login_window)
        )
        device_changes = sum(
            record.device_changed and since(record, self._device_change_window)
            for record in user_records
        )

        # Preserve insertion order as the tie-breaker for multiple events with
        # the same timestamp (common in deterministic replays and bursts).
        indexed_device_records = [
            (index, record)
            for index, record in enumerate(user_records)
            if record.device_id is not None
        ]
        latest = max(
            indexed_device_records,
            key=lambda item: (item[1].occurred_at, item[0]),
            default=None,
        )
        last_device_record = latest[1] if latest is not None else None
        last_device = last_device_record.device_id if last_device_record else None
        current_device_changed = bool(
            device_id is not None and last_device is not None and device_id != last_device
        )

        def distinct_users(attribute: str, value: str | None) -> int:
            if value is None:
                return 0
            return len(
                {
                    record.user_id
                    for record in visible
                    if record.user_id is not None and getattr(record, attribute) == value
                }
            )

        return StateSnapshot(
            as_of=as_of,
            applications_last_hour=applications_last_hour,
            applications_last_day=applications_last_day,
            login_frequency_24h=login_frequency,
            failed_login_attempts_24h=failed_logins,
            device_changes_30d=device_changes,
            shared_device_user_count=distinct_users("device_id", device_id),
            shared_ip_user_count=distinct_users("ip_address", ip_address),
            shared_bank_user_count=distinct_users("bank_account_id", bank_id),
            device_changed=current_device_changed,
            last_device_id=last_device,
        )

    def _record_locked(self, event: Any, occurred_at: datetime) -> None:
        user_id = self._identifier(event, "user_id", "customer_id", "applicant_id")
        device_id = self._identifier(event, "device_id", "device_fingerprint")
        prior_devices = (
            [
                record
                for record in self._events
                if record.user_id == user_id
                and record.device_id is not None
                and record.occurred_at <= occurred_at
            ]
            if user_id
            else []
        )
        last = max(
            enumerate(prior_devices),
            key=lambda item: (item[1].occurred_at, item[0]),
            default=None,
        )
        last_record = last[1] if last is not None else None
        changed = bool(last_record and device_id is not None and device_id != last_record.device_id)
        kind = self._event_kind(event)
        failed_count = self._failed_login_count(event, kind)
        self._events.append(
            _RecordedEvent(
                occurred_at=occurred_at,
                user_id=user_id,
                device_id=device_id,
                ip_address=self._identifier(event, "ip_address", "ip"),
                bank_account_id=self._identifier(
                    event, "bank_account_id", "bank_id", "bank_account"
                ),
                event_kind=kind,
                failed_login_count=failed_count,
                device_changed=changed,
            )
        )

    def _prune_locked(self, as_of: datetime) -> int:
        cutoff = as_of - self._ttl
        previous = len(self._events)
        # Future-dated entries are retained but remain invisible to snapshots.
        self._events[:] = [record for record in self._events if record.occurred_at >= cutoff]
        return previous - len(self._events)

    def _resolve_time(self, event: Any, explicit: datetime | None) -> datetime:
        if explicit is not None:
            return self._normalise_datetime(explicit)
        value = self._value(event, "occurred_at", "event_timestamp", "timestamp", "created_at")
        if value is _MISSING or value is None:
            value = self._clock()
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime):
            raise TypeError("event timestamp must be a datetime or ISO-8601 string")
        return self._normalise_datetime(value)

    @staticmethod
    def _normalise_datetime(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            # Treat fixture/replay timestamps without an offset as UTC while
            # keeping every stored comparison timezone-aware.
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @classmethod
    def _identifier(cls, event: Any, *names: str) -> str | None:
        value = cls._value(event, *names)
        if value is _MISSING or value is None or value == "":
            return None
        if isinstance(value, Enum):
            value = value.value
        return str(value)

    @classmethod
    def _event_kind(cls, event: Any) -> str:
        value = cls._value(event, "event_type", "event_kind", "type")
        if value is _MISSING or value is None:
            return "application"
        if isinstance(value, Enum):
            value = value.value
        normalised = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if normalised in {"application", "loan_application", "apply", "transaction"}:
            return "application"
        if normalised in {"failed_login", "login_failed", "authentication_failure"}:
            return "failed_login"
        if normalised in {"login", "successful_login", "login_success"}:
            return "login"
        return normalised

    @classmethod
    def _failed_login_count(cls, event: Any, kind: str) -> int:
        value = cls._value(event, "failed_login_count", "attempt_count")
        if value is _MISSING or value is None:
            return 1 if kind == "failed_login" else 0
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("failed login count must be a non-negative integer") from exc
        if count < 0:
            raise ValueError("failed login count must be a non-negative integer")
        return count if kind == "failed_login" else 0

    @staticmethod
    def _value(event: Any, *names: str) -> Any:
        if isinstance(event, Mapping):
            for name in names:
                if name in event:
                    return event[name]
            return _MISSING
        for name in names:
            if hasattr(event, name):
                return getattr(event, name)
        return _MISSING
