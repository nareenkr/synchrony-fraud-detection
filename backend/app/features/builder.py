"""Deterministic feature builder used by both training replay and inference."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from math import cos, isfinite, pi, sin
from typing import Any

import numpy as np
import pandas as pd

from backend.app.schemas import LoanApplicationEvent
from backend.app.state import StateSnapshot

from .contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    validate_feature_frame,
    validate_feature_vector,
)


class FeatureValidationError(ValueError):
    """Raised when an event or state value cannot safely become a feature."""


_MISSING = object()


def _read(source: object, *names: str, default: Any = _MISSING) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    else:
        for name in names:
            if hasattr(source, name):
                return getattr(source, name)
    if default is not _MISSING:
        return default
    raise FeatureValidationError(f"required field {names[0]!r} is missing")


def _number(
    source: object,
    *names: str,
    default: float | None = None,
    nonnegative: bool = False,
) -> float:
    value = _read(source, *names, default=default)
    if value is None:
        if default is None:
            raise FeatureValidationError(f"required field {names[0]!r} is missing")
        value = default
    if isinstance(value, bool):
        result = float(value)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
    else:
        raise FeatureValidationError(f"field {names[0]!r} must be numeric")
    if not isfinite(result):
        raise FeatureValidationError(f"field {names[0]!r} must be finite")
    if nonnegative and result < 0.0:
        raise FeatureValidationError(f"field {names[0]!r} cannot be negative")
    return result


def _flag(source: object, *names: str, default: bool = False) -> float:
    value = _read(source, *names, default=default)
    if value in (True, 1, 1.0):
        return 1.0
    if value in (False, 0, 0.0, None):
        return 0.0
    raise FeatureValidationError(f"field {names[0]!r} must be boolean")


def _clip(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _safe_ratio(numerator: float, denominator: float, maximum: float) -> float:
    if denominator > 0.0:
        return _clip(numerator / denominator, 0.0, maximum)
    return maximum if numerator > 0.0 else 0.0


def _timestamp(event: object) -> datetime:
    value = _read(event, "event_timestamp", "occurred_at", "submitted_at", "timestamp")
    if not isinstance(value, datetime):
        raise FeatureValidationError("event_timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise FeatureValidationError("event_timestamp must include a timezone")
    return value


class FeatureBuilder:
    """Build canonical features from a validated event and *prior* state.

    ``snapshot`` must be captured before the current event is recorded. This
    ordering is what keeps both online inference and chronological replay from
    self-counting the application.
    """

    schema_version = FEATURE_SCHEMA_VERSION
    feature_names = FEATURE_NAMES

    def transform(
        self,
        event: LoanApplicationEvent | Mapping[str, Any],
        snapshot: StateSnapshot | Mapping[str, Any],
    ) -> pd.DataFrame:
        """Return a validated one-row float64 DataFrame in model order."""

        values = self._values(event, snapshot)
        frame = pd.DataFrame([values], columns=FEATURE_NAMES, dtype=np.float64)
        try:
            return validate_feature_frame(frame)
        except ValueError as exc:
            raise FeatureValidationError(str(exc)) from exc

    def transform_vector(
        self,
        event: LoanApplicationEvent | Mapping[str, Any],
        snapshot: StateSnapshot | Mapping[str, Any],
    ) -> np.ndarray:
        """Return the same features as a validated one-dimensional vector."""

        return validate_feature_vector(self._values(event, snapshot)).copy()

    def _values(self, event: object, snapshot: object) -> list[float]:
        amount = _number(
            event,
            "requested_loan_amount",
            "amount",
            nonnegative=True,
        )
        transaction_amount = _number(event, "transaction_amount", default=amount, nonnegative=True)
        income = _number(event, "income", "annual_income", default=0.0, nonnegative=True)
        debt_to_income = _number(event, "debt_to_income_ratio", default=0.0, nonnegative=True)
        account_age = _number(event, "account_age_days", default=0.0, nonnegative=True)
        bank_age = _number(event, "bank_account_age_days", default=0.0, nonnegative=True)
        balance_before = _number(
            event, "origin_balance_before", "oldbalance_org", default=0.0, nonnegative=True
        )
        balance_after = _number(
            event,
            "origin_balance_after",
            "newbalance_org",
            default=balance_before,
            nonnegative=True,
        )

        applications_hour = _number(
            snapshot,
            "applications_last_hour",
            "applications_last_1h",
            default=0.0,
            nonnegative=True,
        )
        applications_day = _number(
            snapshot,
            "applications_last_day",
            "applications_last_24h",
            default=0.0,
            nonnegative=True,
        )
        transaction_frequency = _number(
            snapshot,
            "transaction_frequency_24h",
            default=_number(
                event,
                "transaction_frequency_24h",
                "transaction_frequency",
                default=applications_day,
                nonnegative=True,
            ),
            nonnegative=True,
        )

        explicit_deviation = _read(event, "transaction_amount_deviation", default=None)
        if explicit_deviation is None:
            prior_mean = _number(snapshot, "prior_amount_mean", default=0.0, nonnegative=True)
            prior_std = _number(snapshot, "prior_amount_std", default=0.0, nonnegative=True)
            prior_count = _number(snapshot, "prior_amount_count", default=0.0, nonnegative=True)
            deviation = (
                (transaction_amount - prior_mean) / prior_std
                if prior_count >= 2.0 and prior_std > 0.0
                else 0.0
            )
        else:
            deviation = _number(event, "transaction_amount_deviation")

        # Supplied trailing observations and locally retained state can cover
        # different channels. Taking their maximum avoids both discarding a
        # supplied observation and double-counting overlapping windows.
        device_changes = max(
            _number(snapshot, "device_changes_30d", default=0.0, nonnegative=True),
            _number(event, "device_changes_30d", default=0.0, nonnegative=True),
        )
        login_frequency = max(
            _number(snapshot, "login_frequency_24h", default=0.0, nonnegative=True),
            _number(event, "login_frequency_24h", default=0.0, nonnegative=True),
        )
        failed_logins = max(
            _number(snapshot, "failed_login_attempts_24h", default=0.0, nonnegative=True),
            _number(event, "failed_login_attempts_24h", default=0.0, nonnegative=True),
        )
        previous_rejected = _number(
            event,
            "previous_rejected_applications",
            default=_number(
                snapshot, "previous_rejected_applications", default=0.0, nonnegative=True
            ),
            nonnegative=True,
        )

        last_device_id = _read(snapshot, "last_device_id", default=None)
        device_changed = _flag(snapshot, "device_changed", default=False)
        explicit_new_device = _read(event, "is_new_device", default=None)
        if explicit_new_device is not None:
            is_new_device = _flag(event, "is_new_device")
        else:
            current_device_id = _read(event, "device_id", default=None)
            is_new_device = float(
                current_device_id is not None and (last_device_id is None or bool(device_changed))
            )

        shared_device = _number(
            snapshot,
            "shared_device_user_count",
            "distinct_users_for_device",
            default=_number(event, "shared_device_user_count", default=0.0, nonnegative=True),
            nonnegative=True,
        )
        shared_ip = _number(
            snapshot,
            "shared_ip_user_count",
            "distinct_users_for_ip",
            default=_number(event, "shared_ip_user_count", default=0.0, nonnegative=True),
            nonnegative=True,
        )
        shared_bank = _number(
            snapshot,
            "shared_bank_user_count",
            "distinct_users_for_bank",
            default=0.0,
            nonnegative=True,
        )

        occurred_at = _timestamp(event)
        hour = occurred_at.hour + occurred_at.minute / 60.0 + occurred_at.second / 3600.0
        usual_hour_raw = _read(snapshot, "usual_login_hour", default=None)
        if usual_hour_raw is None:
            usual_hour_raw = _read(event, "usual_login_hour", default=None)
        if usual_hour_raw is None:
            hour_deviation = _number(event, "hour_of_day_deviation", default=0.0, nonnegative=True)
        else:
            usual_hour = _number({"usual_hour": usual_hour_raw}, "usual_hour", nonnegative=True)
            if usual_hour > 24.0:
                raise FeatureValidationError("usual_login_hour must be at most 24")
            difference = abs(hour - (usual_hour % 24.0))
            hour_deviation = min(difference, 24.0 - difference)

        raw = {
            "requested_loan_amount": amount,
            "income": income,
            "loan_to_income_ratio": _safe_ratio(amount, income, 100.0),
            "debt_to_income_ratio": debt_to_income,
            "account_age_days": account_age,
            "bank_account_age_days": bank_age,
            "origin_balance_before": balance_before,
            "amount_to_balance_ratio": _safe_ratio(transaction_amount, balance_before, 100.0),
            "balance_change_ratio": _clip(
                (balance_after - balance_before) / max(balance_before, 1.0), -100.0, 100.0
            ),
            "applications_last_hour": applications_hour,
            "applications_last_day": applications_day,
            "transaction_frequency_24h": transaction_frequency,
            "transaction_amount_deviation": _clip(deviation, -20.0, 20.0),
            "device_changes_30d": device_changes,
            "login_frequency_24h": login_frequency,
            "failed_login_attempts_24h": failed_logins,
            "previous_rejected_applications": previous_rejected,
            "is_new_device": is_new_device,
            "unusual_login_location": _flag(event, "unusual_login_location", default=False),
            "shared_device_user_count": shared_device,
            "shared_ip_user_count": shared_ip,
            "shared_bank_user_count": shared_bank,
            "hour_sin": sin(2.0 * pi * hour / 24.0),
            "hour_cos": cos(2.0 * pi * hour / 24.0),
            "hour_of_day_deviation": _clip(hour_deviation, 0.0, 12.0),
            "is_night_application": float(occurred_at.hour < 6),
        }

        # Saturate legitimate high-volume/extreme values to the explicit model
        # domain. Invalid negatives and non-finite inputs were rejected above.
        bounded = {
            spec.name: _clip(raw[spec.name], spec.minimum, spec.maximum) for spec in FEATURE_SPECS
        }
        return [bounded[name] for name in FEATURE_NAMES]
