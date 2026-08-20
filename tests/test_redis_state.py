"""Redis state adapter tests using a deterministic, in-process Redis model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from fnmatch import fnmatch
from threading import Barrier, Lock, Thread
from typing import Any

import pytest

from backend.app.state import RealtimeStateStore, RedisRealtimeStateStore


class FakeRedis:
    """Small model of the commands and Lua contract used by the adapter."""

    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, float]] = {}
        self.sequences: dict[str, int] = {}
        self.eval_keys: list[list[str]] = []
        self.eval_args: list[list[Any]] = []
        self._lock = Lock()

    def eval(self, _script: str, key_count: int, *values: Any) -> list[Any]:
        keys = [str(value) for value in values[:key_count]]
        args = list(values[key_count:])
        with self._lock:
            self.eval_keys.append(keys)
            self.eval_args.append(args)
            as_of, cutoff, hour, day, login, change = map(float, args[:6])
            kind = str(args[6])
            failed = int(args[7])
            device, device_value, user = map(str, args[8:11])
            should_record = args[11] == "1"
            nonce = str(args[12])
            ip, bank = str(args[14]), str(args[15])

            for key in keys[:5]:
                self._remove(key, lambda score: score < cutoff)
            records = (
                [member for member, _score in self._range(keys[0], cutoff, as_of)] if user else []
            )
            apps_hour = sum(r["kind"] == "application" and r["time"] >= hour for r in records)
            apps_day = sum(r["kind"] == "application" and r["time"] >= day for r in records)
            logins = sum(
                r["kind"] in {"login", "failed_login"} and r["time"] >= login for r in records
            )
            failures = sum(r["failed"] for r in records if r["time"] >= login)
            changes = sum(r["changed"] and r["time"] >= change for r in records)
            device_records = [r for r in records if r["device"]]
            last = max(device_records, key=lambda r: (r["time"], r["sequence"]), default=None)
            last_token = last["device"] if last else ""
            last_value = last["device_value"] if last else ""
            current_changed = bool(user and device and last_token and device != last_token)

            def distinct(key: str, present: str) -> int:
                if not present:
                    return 0
                return len(
                    {
                        record["user"]
                        for record, _score in self._range(key, cutoff, as_of)
                        if record["user"]
                    }
                )

            result = [
                apps_hour,
                apps_day,
                logins,
                failures,
                changes,
                distinct(keys[1], device),
                distinct(keys[2], ip),
                distinct(keys[3], bank),
                int(current_changed),
                last_value.encode(),
            ]
            if should_record:
                sequence = self.sequences.get(keys[5], 0) + 1
                self.sequences[keys[5]] = sequence
                user_record = {
                    "sequence": sequence,
                    "time": as_of,
                    "kind": kind,
                    "failed": failed,
                    "device": device,
                    "device_value": device_value,
                    "changed": current_changed,
                    "nonce": nonce,
                }
                if user:
                    self._add(keys[0], user_record, as_of)
                entity_record = {"sequence": sequence, "user": user, "nonce": nonce}
                for key, present in zip(keys[1:4], (device, ip, bank), strict=True):
                    if present:
                        self._add(key, entity_record.copy(), as_of)
                self._add(keys[4], {"sequence": sequence, "nonce": nonce}, as_of)
            return result

    def zremrangebyscore(self, key: str, _minimum: str, maximum: str) -> int:
        cutoff = float(maximum.removeprefix("("))
        with self._lock:
            return self._remove(key, lambda score: score < cutoff)

    def scan(self, *, cursor: int | str | bytes, match: str, count: int) -> tuple[int, list[str]]:
        del cursor, count
        keys = set(self.zsets) | set(self.sequences)
        return 0, sorted(key for key in keys if fnmatch(key, match))

    def unlink(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += int(self.zsets.pop(key, None) is not None)
            removed += int(self.sequences.pop(key, None) is not None)
        return removed

    def _add(self, key: str, member: dict[str, Any], score: float) -> None:
        identity = f"{member['sequence']:020}|{member.get('nonce', '')}"
        self.zsets.setdefault(key, {})[identity] = score
        # Keep structured members separately without trying to execute Lua.
        self.zsets[key][identity] = score
        setattr(self, f"record_{key}_{identity}", member)

    def _range(
        self, key: str, minimum: float, maximum: float
    ) -> list[tuple[dict[str, Any], float]]:
        values = []
        for identity, score in self.zsets.get(key, {}).items():
            if minimum <= score <= maximum:
                values.append((getattr(self, f"record_{key}_{identity}"), score))
        return sorted(values, key=lambda item: (item[1], item[0]["sequence"]))

    def _remove(self, key: str, predicate: Any) -> int:
        target = self.zsets.get(key, {})
        identities = [identity for identity, score in target.items() if predicate(score)]
        for identity in identities:
            del target[identity]
            delattr(self, f"record_{key}_{identity}")
        return len(identities)


NOW = datetime(2026, 8, 20, 12, tzinfo=UTC)
SECRET = "redis-state-test-secret-long-enough"


def event(user: str, **overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "user_id": user,
        "device_id": f"device-{user}",
        "ip_address": f"10.2.0.{len(user)}",
        "bank_account_id": f"bank-{user}",
    }
    result.update(overrides)
    return result


def store(client: FakeRedis, namespace: str = "demo") -> RedisRealtimeStateStore:
    return RedisRealtimeStateStore(
        client,
        identifier_secret=SECRET,
        namespace=namespace,
        clock=lambda: NOW,
    )


def test_atomic_snapshot_is_prior_only_under_concurrency() -> None:
    state = store(FakeRedis())
    barrier = Barrier(8)
    observed: list[int] = []

    def worker() -> None:
        barrier.wait()
        observed.append(state.snapshot_and_record(event("raw-user")).applications_last_hour)

    threads = [Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(observed) == list(range(8))
    assert isinstance(state, RealtimeStateStore)


def test_windows_distinct_users_logins_and_device_changes_match_contract() -> None:
    state = store(FakeRedis())
    shared = {
        "device_id": "shared-device",
        "ip_address": "shared-ip",
        "bank_account_id": "shared-bank",
    }
    state.record_event(event("u1", **shared), now=NOW - timedelta(hours=1))
    state.record_event(event("u1", event_type="login"), now=NOW - timedelta(hours=2))
    state.record_event(
        event("u1", event_type="failed-login", failed_login_count=3),
        now=NOW - timedelta(hours=3),
    )
    changed = state.snapshot_and_record(event("u1", device_id="new-device"), now=NOW)
    before_u2 = state.snapshot_and_record(event("u2", **shared), now=NOW)
    after = state.get_snapshot(event("u3", **shared), now=NOW)

    assert changed.applications_last_hour == 1  # inclusive boundary
    assert changed.applications_last_day == 1
    assert changed.login_frequency_24h == 2
    assert changed.failed_login_attempts_24h == 3
    assert changed.device_changed is True
    assert before_u2.shared_device_user_count == 1
    assert before_u2.shared_ip_user_count == 1
    assert before_u2.shared_bank_user_count == 1
    assert after.shared_device_user_count == 2
    assert state.get_snapshot(event("u1", device_id="new-device")).device_changes_30d == 1


def test_future_events_are_invisible_and_ttl_prune_counts_global_events() -> None:
    client = FakeRedis()
    state = store(client)
    state.record_event(event("u1"), now=NOW + timedelta(hours=1))
    state.record_event(event("old"), now=NOW - timedelta(days=30, seconds=1))

    assert state.prune(now=NOW) == 1
    assert state.get_snapshot(event("u1"), now=NOW).applications_last_day == 0


def test_events_without_users_do_not_share_velocity_or_device_history() -> None:
    state = store(FakeRedis())
    anonymous = {"device_id": "shared-device"}

    first = state.snapshot_and_record(anonymous)
    second = state.snapshot_and_record(anonymous)

    assert first.applications_last_day == second.applications_last_day == 0
    assert first.device_changed is second.device_changed is False
    assert second.shared_device_user_count == 0


def test_keys_are_hmac_pseudonyms_and_reset_is_exactly_namespaced() -> None:
    client = FakeRedis()
    demo = store(client, "demo")
    sibling = store(client, "demo-other")
    raw_values = ("raw-user-991", "raw-device-882", "192.0.2.77", "raw-bank-773")
    fields = ("user_id", "device_id", "ip_address", "bank_account_id")
    payload = dict(zip(fields, raw_values, strict=True))
    demo.record_event(payload)
    sibling.record_event(payload)

    demo_keys = client.eval_keys[0]
    assert all(raw not in key for raw in raw_values for key in demo_keys)
    assert all(raw not in str(argument) for raw in raw_values for argument in client.eval_args[0])
    assert all(key.startswith("synchrony:{demo}:state:") for key in demo_keys)
    sibling_key = next(key for key in client.zsets if key.startswith("synchrony:{demo-other}"))

    demo.reset()

    assert sibling_key in client.zsets
    assert not any(key.startswith("synchrony:{demo}:state:") for key in client.zsets)


def test_configuration_validation() -> None:
    with pytest.raises(ValueError, match="namespace"):
        RedisRealtimeStateStore(FakeRedis(), identifier_secret=SECRET, namespace="unsafe:*")
    with pytest.raises(ValueError, match="must cover"):
        RedisRealtimeStateStore(
            FakeRedis(),
            identifier_secret=SECRET,
            ttl=timedelta(hours=1),
        )
