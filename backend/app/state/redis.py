"""Redis implementation of the real-time behavioral state contract.

The adapter keeps all identifiers out of Redis keys by applying a keyed HMAC
pseudonym before constructing them.  Its Lua operation computes a prior-only
snapshot and, when requested, appends the current event in the same atomic
server-side operation.  This makes the store safe to share between API
workers while retaining the point-in-time behavior of the memory adapter.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from backend.app.core.privacy import Pseudonymizer

from .base import StateSnapshot

_MISSING = object()
_NAMESPACE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

# KEYS: user events, device users, IP users, bank users, all events, sequence.
# Every data member begins with a zero-padded sequence. This preserves insertion
# order when multiple events have the same occurrence timestamp.
_SNAPSHOT_AND_RECORD_LUA = r"""
local asof = tonumber(ARGV[1])
local cutoff = tonumber(ARGV[2])
local hour_start = tonumber(ARGV[3])
local day_start = tonumber(ARGV[4])
local login_start = tonumber(ARGV[5])
local change_start = tonumber(ARGV[6])
local kind = ARGV[7]
local failed = tonumber(ARGV[8])
local device_token = ARGV[9]
local device_value = ARGV[10]
local user_token = ARGV[11]
local should_record = ARGV[12] == '1'
local event_nonce = ARGV[13]
local ttl_seconds = tonumber(ARGV[14])

for index = 1, 5 do
  redis.call('ZREMRANGEBYSCORE', KEYS[index], '-inf', '(' .. cutoff)
end

local user_events = {}
if user_token ~= '' then
  user_events = redis.call('ZRANGEBYSCORE', KEYS[1], cutoff, asof)
end
local apps_hour = 0
local apps_day = 0
local logins = 0
local failed_logins = 0
local changes = 0
local last_score = nil
local last_sequence = nil
local last_device_token = ''
local last_device_value = ''

for _, member in ipairs(user_events) do
  local separator = string.find(member, '|', 1, true)
  local sequence = tonumber(string.sub(member, 1, separator - 1))
  local record = cjson.decode(string.sub(member, separator + 1))
  local occurred = tonumber(record['t'])
  if record['k'] == 'application' then
    if occurred >= hour_start then apps_hour = apps_hour + 1 end
    if occurred >= day_start then apps_day = apps_day + 1 end
  end
  if (record['k'] == 'login' or record['k'] == 'failed_login') and occurred >= login_start then
    logins = logins + 1
  end
  if occurred >= login_start then failed_logins = failed_logins + tonumber(record['f']) end
  if occurred >= change_start and record['c'] then changes = changes + 1 end
  if record['d'] ~= '' and
     (last_score == nil or occurred > last_score or
      (occurred == last_score and sequence > last_sequence)) then
    last_score = occurred
    last_sequence = sequence
    last_device_token = record['d']
    last_device_value = record['v']
  end
end

local function distinct_users(key)
  local seen = {}
  local count = 0
  local members = redis.call('ZRANGEBYSCORE', key, cutoff, asof)
  for _, member in ipairs(members) do
    local separator = string.find(member, '|', 1, true)
    local record = cjson.decode(string.sub(member, separator + 1))
    local user = record['u']
    if user ~= '' and not seen[user] then
      seen[user] = true
      count = count + 1
    end
  end
  return count
end

local device_users = device_token ~= '' and distinct_users(KEYS[2]) or 0
local ip_users = ARGV[15] ~= '' and distinct_users(KEYS[3]) or 0
local bank_users = ARGV[16] ~= '' and distinct_users(KEYS[4]) or 0
local current_changed = user_token ~= '' and device_token ~= '' and last_device_token ~= '' and
                        device_token ~= last_device_token

if should_record then
  local sequence = redis.call('INCR', KEYS[6])
  local prefix = string.format('%020d', sequence) .. '|'
  local user_record = cjson.encode({t=asof, k=kind, f=failed,
                                    d=device_token, v=device_value,
                                    c=current_changed})
  if user_token ~= '' then redis.call('ZADD', KEYS[1], asof, prefix .. user_record) end
  local entity_record = prefix .. cjson.encode({u=user_token, n=event_nonce})
  if device_token ~= '' then redis.call('ZADD', KEYS[2], asof, entity_record) end
  if ARGV[15] ~= '' then redis.call('ZADD', KEYS[3], asof, entity_record) end
  if ARGV[16] ~= '' then redis.call('ZADD', KEYS[4], asof, entity_record) end
  redis.call('ZADD', KEYS[5], asof, prefix .. event_nonce)
  for index = 1, 6 do redis.call('EXPIRE', KEYS[index], ttl_seconds) end
end

return {apps_hour, apps_day, logins, failed_logins, changes,
        device_users, ip_users, bank_users, current_changed and 1 or 0,
        last_device_value}
"""


class RedisRealtimeStateStore:
    """Atomic Redis-backed behavioral state with bounded occurrence windows.

    ``client`` is intentionally injectable so callers can provide a configured
    redis-py client and tests do not need a live Redis server. ``identifier_secret``
    must be the application's configured pseudonym/HMAC secret.
    """

    def __init__(
        self,
        client: Any,
        *,
        identifier_secret: str | bytes,
        namespace: str = "default",
        clock: Callable[[], datetime] | None = None,
        ttl: timedelta = timedelta(days=30),
        application_hour_window: timedelta = timedelta(hours=1),
        application_day_window: timedelta = timedelta(days=1),
        login_window: timedelta = timedelta(days=1),
        device_change_window: timedelta = timedelta(days=30),
    ) -> None:
        clean_namespace = namespace.strip().lower()
        if not _NAMESPACE.fullmatch(clean_namespace):
            raise ValueError("namespace must contain only lowercase letters, digits, '_' or '-'")
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

        self._client = client
        self._pseudonymizer = Pseudonymizer(identifier_secret)
        self._namespace = clean_namespace
        # Hash tags keep every script key in one Redis Cluster slot.
        self._prefix = f"synchrony:{{{clean_namespace}}}:state:"
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ttl = ttl
        self._application_hour_window = application_hour_window
        self._application_day_window = application_day_window
        self._login_window = login_window
        self._device_change_window = device_change_window

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> RedisRealtimeStateStore:
        """Build an adapter from a Redis URL without making redis a core dependency."""
        try:
            from redis import Redis
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Redis state requires installation with the 'redis' extra") from exc
        return cls(Redis.from_url(url, decode_responses=True), **kwargs)

    def get_snapshot(self, event: Any, *, now: datetime | None = None) -> StateSnapshot:
        return self._execute(event, now=now, record=False)

    def snapshot(self, event: Any, *, now: datetime | None = None) -> StateSnapshot:
        return self.get_snapshot(event, now=now)

    def record_event(self, event: Any, *, now: datetime | None = None) -> None:
        self._execute(event, now=now, record=True)

    def record(self, event: Any, *, now: datetime | None = None) -> None:
        self.record_event(event, now=now)

    def snapshot_and_record(self, event: Any, *, now: datetime | None = None) -> StateSnapshot:
        return self._execute(event, now=now, record=True)

    def prune(self, *, now: datetime | None = None) -> int:
        as_of = self._normalise_datetime(now if now is not None else self._clock())
        cutoff = as_of.timestamp() - self._ttl.total_seconds()
        return int(self._client.zremrangebyscore(self._prefix + "events", "-inf", f"({cutoff}"))

    def reset(self) -> None:
        """Delete only keys belonging to this exact validated namespace."""
        cursor: int | str | bytes = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=self._prefix + "*", count=500)
            if keys:
                # UNLINK avoids blocking Redis for a namespace containing many keys.
                unlink = getattr(self._client, "unlink", None)
                (unlink or self._client.delete)(*keys)
            if int(cursor) == 0:
                break

    def ping(self) -> bool:
        """Check connectivity during explicit Redis-mode startup/readiness."""

        return bool(self._client.ping())

    def close(self) -> None:
        """Release the redis-py connection pool when the application stops."""

        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _execute(self, event: Any, *, now: datetime | None, record: bool) -> StateSnapshot:
        as_of = self._resolve_time(event, now)
        user = self._identifier(event, "user_id", "customer_id", "applicant_id")
        device = self._identifier(event, "device_id", "device_fingerprint")
        ip_address = self._identifier(event, "ip_address", "ip")
        bank = self._identifier(event, "bank_account_id", "bank_id", "bank_account")
        user_token = self._token("user", user)
        device_token = self._token("device", device)
        ip_token = self._token("ip", ip_address)
        bank_token = self._token("bank", bank)
        second = as_of.timestamp()
        keys = [
            self._entity_key("user", user_token),
            self._entity_key("device", device_token),
            self._entity_key("ip", ip_token),
            self._entity_key("bank", bank_token),
            self._prefix + "events",
            self._prefix + "sequence",
        ]
        args = [
            second,
            second - self._ttl.total_seconds(),
            second - self._application_hour_window.total_seconds(),
            second - self._application_day_window.total_seconds(),
            second - self._login_window.total_seconds(),
            second - self._device_change_window.total_seconds(),
            self._event_kind(event),
            self._failed_login_count(event),
            device_token,
            # Redis never needs the raw device identifier.  The keyed token is
            # enough to report prior-device context and compare transitions.
            device_token,
            user_token,
            "1" if record else "0",
            uuid.uuid4().hex,
            math.ceil(self._ttl.total_seconds()),
            ip_token,
            bank_token,
        ]
        values = self._client.eval(_SNAPSHOT_AND_RECORD_LUA, len(keys), *keys, *args)
        decoded = [value.decode() if isinstance(value, bytes) else value for value in values]
        return StateSnapshot(
            as_of=as_of,
            applications_last_hour=int(decoded[0]),
            applications_last_day=int(decoded[1]),
            login_frequency_24h=int(decoded[2]),
            failed_login_attempts_24h=int(decoded[3]),
            device_changes_30d=int(decoded[4]),
            shared_device_user_count=int(decoded[5]),
            shared_ip_user_count=int(decoded[6]),
            shared_bank_user_count=int(decoded[7]),
            device_changed=bool(decoded[8]),
            last_device_id=str(decoded[9]) or None,
        )

    def _entity_key(self, kind: str, token: str) -> str:
        return f"{self._prefix}{kind}:{token or 'none'}"

    def _token(self, kind: str, value: str | None) -> str:
        if value is None:
            return ""
        return self._pseudonymizer.pseudonymize(f"state_{self._namespace}_{kind}", value)

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
    def _failed_login_count(cls, event: Any) -> int:
        kind = cls._event_kind(event)
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


__all__ = ["RedisRealtimeStateStore"]
