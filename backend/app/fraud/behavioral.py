"""Transparent behavioral and velocity fraud scoring.

The scorer consumes the canonical numeric feature contract.  It deliberately
does not inspect identifiers or raw locations, which keeps emitted signals safe
to persist and display.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from backend.app.features.contract import FEATURE_NAMES, validate_feature_vector
from backend.app.schemas import RiskSignal


@dataclass(frozen=True, slots=True)
class BehavioralConfig:
    """Thresholds and relative weights for each behavioral family."""

    applications_hour_low: float = 1.0
    applications_hour_high: float = 6.0
    applications_day_low: float = 3.0
    applications_day_high: float = 15.0
    device_changes_low: float = 1.0
    device_changes_high: float = 6.0
    new_device_account_age_days: float = 90.0
    loan_ratio_low: float = 0.35
    loan_ratio_high: float = 2.0
    failed_logins_low: float = 1.0
    failed_logins_high: float = 8.0
    amount_deviation_low: float = 1.5
    amount_deviation_high: float = 8.0
    transaction_frequency_low: float = 10.0
    transaction_frequency_high: float = 80.0
    loan_amount_low: float = 15_000.0
    loan_amount_high: float = 100_000.0
    burst_weight: float = 1.4
    device_weight: float = 1.0
    loan_ratio_weight: float = 1.2
    failed_login_weight: float = 1.0
    unusual_location_weight: float = 0.8
    amount_deviation_weight: float = 1.0
    transaction_volume_weight: float = 0.8
    large_request_weight: float = 0.8

    def __post_init__(self) -> None:
        ranges = (
            (self.applications_hour_low, self.applications_hour_high),
            (self.applications_day_low, self.applications_day_high),
            (self.device_changes_low, self.device_changes_high),
            (self.loan_ratio_low, self.loan_ratio_high),
            (self.failed_logins_low, self.failed_logins_high),
            (self.amount_deviation_low, self.amount_deviation_high),
            (self.transaction_frequency_low, self.transaction_frequency_high),
            (self.loan_amount_low, self.loan_amount_high),
        )
        if any(low < 0 or high <= low for low, high in ranges):
            raise ValueError("behavioral thresholds must be non-negative and strictly ordered")
        if self.new_device_account_age_days <= 0:
            raise ValueError("new-device account-age threshold must be positive")
        weights = self.weights().values()
        if any(weight < 0 for weight in weights) or sum(self.weights().values()) <= 0:
            raise ValueError("behavioral weights must be non-negative with a positive sum")

    def weights(self) -> dict[str, float]:
        return {
            "burst": self.burst_weight,
            "device": self.device_weight,
            "loan_ratio": self.loan_ratio_weight,
            "failed_logins": self.failed_login_weight,
            "unusual_location": self.unusual_location_weight,
            "amount_deviation": self.amount_deviation_weight,
            "transaction_volume": self.transaction_volume_weight,
            "large_request": self.large_request_weight,
        }


DEFAULT_BEHAVIORAL_CONFIG = BehavioralConfig()


@dataclass(frozen=True, slots=True)
class BehavioralScore:
    """Normalized component output and human-readable supporting evidence."""

    risk: float
    signals: tuple[RiskSignal, ...]

    @property
    def score(self) -> float:
        return self.risk


def _ramp(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _feature_values(
    features: Mapping[str, float] | pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
) -> dict[str, float]:
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("behavioral scoring requires exactly one feature row")
        if tuple(features.columns) != FEATURE_NAMES:
            raise ValueError("feature columns do not match the ordered feature contract")
        vector = validate_feature_vector(features.iloc[0].to_numpy(dtype=float))
    elif isinstance(features, (Mapping, pd.Series)):
        missing = [name for name in FEATURE_NAMES if name not in features]
        if missing:
            raise ValueError(f"missing canonical features: {', '.join(missing)}")
        vector = validate_feature_vector([features[name] for name in FEATURE_NAMES])
    else:
        vector = validate_feature_vector(features)
    return dict(zip(FEATURE_NAMES, vector, strict=True))


class BehavioralScorer:
    """Calculate a deterministic weighted behavioral risk in ``[0, 1]``."""

    def __init__(self, config: BehavioralConfig = DEFAULT_BEHAVIORAL_CONFIG) -> None:
        self.config = config

    def score(
        self,
        features: Mapping[str, float] | pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    ) -> BehavioralScore:
        values = _feature_values(features)
        config = self.config

        hour_burst = _ramp(
            values["applications_last_hour"],
            config.applications_hour_low,
            config.applications_hour_high,
        )
        day_burst = _ramp(
            values["applications_last_day"],
            config.applications_day_low,
            config.applications_day_high,
        )
        burst = max(hour_burst, day_burst)

        changes = _ramp(
            values["device_changes_30d"],
            config.device_changes_low,
            config.device_changes_high,
        )
        # A first-seen device is weak evidence for an established account but
        # strong evidence when paired with a very new account.
        new_device = values["is_new_device"] * _ramp(
            config.new_device_account_age_days - values["account_age_days"],
            0.0,
            config.new_device_account_age_days,
        )
        device = max(changes, new_device)
        loan_ratio = _ramp(
            values["loan_to_income_ratio"], config.loan_ratio_low, config.loan_ratio_high
        )
        failed_logins = _ramp(
            values["failed_login_attempts_24h"],
            config.failed_logins_low,
            config.failed_logins_high,
        )
        unusual_location = float(values["unusual_login_location"])
        amount_deviation = _ramp(
            abs(values["transaction_amount_deviation"]),
            config.amount_deviation_low,
            config.amount_deviation_high,
        )
        transaction_volume = _ramp(
            values["transaction_frequency_24h"],
            config.transaction_frequency_low,
            config.transaction_frequency_high,
        )
        large_request = _ramp(
            values["requested_loan_amount"], config.loan_amount_low, config.loan_amount_high
        )

        factors = {
            "burst": burst,
            "device": device,
            "loan_ratio": loan_ratio,
            "failed_logins": failed_logins,
            "unusual_location": unusual_location,
            "amount_deviation": amount_deviation,
            "transaction_volume": transaction_volume,
            "large_request": large_request,
        }
        weights = config.weights()
        risk = sum(factors[name] * weights[name] for name in factors) / sum(weights.values())
        if not isfinite(risk):  # Defensive: canonical validation should make this unreachable.
            raise ValueError("behavioral risk is not finite")

        signals: list[RiskSignal] = []
        if burst > 0:
            signals.append(
                self._signal(
                    "APPLICATION_VELOCITY",
                    "Recent application velocity is unusually high.",
                    burst,
                )
            )
        if device > 0:
            signals.append(
                self._signal(
                    "RECENT_DEVICE_CHANGE",
                    "Device activity differs from the account's established pattern.",
                    device,
                )
            )
        if loan_ratio > 0:
            signals.append(
                self._signal(
                    "HIGH_LOAN_TO_INCOME",
                    "Requested loan is high relative to declared income.",
                    loan_ratio,
                )
            )
        if failed_logins > 0:
            signals.append(
                self._signal(
                    "REPEATED_FAILED_LOGINS",
                    "Multiple recent login attempts were unsuccessful.",
                    failed_logins,
                )
            )
        if unusual_location > 0:
            signals.append(
                self._signal(
                    "UNUSUAL_LOGIN_LOCATION",
                    "Login location differs from established account behavior.",
                    unusual_location,
                )
            )
        if amount_deviation > 0:
            signals.append(
                self._signal(
                    "AMOUNT_DEVIATION",
                    "Transaction amount differs substantially from prior activity.",
                    amount_deviation,
                )
            )
        if transaction_volume > 0:
            signals.append(
                self._signal(
                    "HIGH_TRANSACTION_VOLUME",
                    "Recent transaction volume is unusually high.",
                    transaction_volume,
                )
            )
        if large_request > 0:
            signals.append(
                self._signal(
                    "LARGE_LOAN_REQUEST",
                    "Requested principal is unusually large.",
                    large_request,
                )
            )
        signals.sort(key=lambda signal: (-signal.severity, signal.code))
        return BehavioralScore(risk=float(np.clip(risk, 0.0, 1.0)), signals=tuple(signals))

    @staticmethod
    def _signal(code: str, message: str, severity: float) -> RiskSignal:
        return RiskSignal(
            code=code,
            message=message,
            severity=float(np.clip(severity, 0.0, 1.0)),
            source="behavioral",
        )
