"""Contracts for point-in-time behavioral state.

The protocol intentionally contains no Redis types.  A Redis adapter can map
these operations to a transaction/Lua script while retaining identical
prior-only semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """State visible immediately before the event being assessed.

    Entity cardinalities are counts of distinct users present in retained
    *prior* events for the entity on the current event.  Consequently a new
    user on an already-seen device observes the existing users, while the
    current user is not added until after scoring.
    """

    as_of: datetime
    applications_last_hour: int = 0
    applications_last_day: int = 0
    login_frequency_24h: int = 0
    failed_login_attempts_24h: int = 0
    device_changes_30d: int = 0
    shared_device_user_count: int = 0
    shared_ip_user_count: int = 0
    shared_bank_user_count: int = 0
    device_changed: bool = False
    last_device_id: str | None = None

    # Explicit aliases make the boundary convenient for callers whose naming
    # follows duration notation rather than the feature provenance catalog.
    @property
    def applications_last_1h(self) -> int:
        return self.applications_last_hour

    @property
    def applications_last_24h(self) -> int:
        return self.applications_last_day

    @property
    def distinct_users_for_device(self) -> int:
        return self.shared_device_user_count

    @property
    def distinct_users_for_ip(self) -> int:
        return self.shared_ip_user_count

    @property
    def distinct_users_for_bank(self) -> int:
        return self.shared_bank_user_count


@runtime_checkable
class RealtimeStateStore(Protocol):
    """Redis-compatible boundary for bounded real-time state.

    ``snapshot_and_record`` must be atomic: it returns a snapshot containing
    only already-recorded events and records the supplied event afterwards.
    """

    def get_snapshot(self, event: Any, *, now: datetime | None = None) -> StateSnapshot: ...

    def record_event(self, event: Any, *, now: datetime | None = None) -> None: ...

    def snapshot_and_record(self, event: Any, *, now: datetime | None = None) -> StateSnapshot: ...

    def prune(self, *, now: datetime | None = None) -> int: ...

    def reset(self) -> None: ...
