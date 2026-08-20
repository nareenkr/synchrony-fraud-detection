"""The immutable model-input contract shared by replay and online scoring.

The schema intentionally contains only numeric, derived values. Identifiers,
timestamps, labels, and raw network/device values must never enter a model
frame. A bundle records ``FEATURE_SCHEMA_VERSION`` and refuses to load when it
does not match this contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd

FEATURE_SCHEMA_VERSION: Final[str] = "lending-fraud-features/v1"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Metadata and inclusive safety bounds for one model feature."""

    name: str
    minimum: float
    maximum: float
    description: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


# Keep this order stable. Appending/reordering requires a schema version bump.
FEATURE_SPECS: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("requested_loan_amount", 0.0, 100_000_000.0, "Requested principal."),
    FeatureSpec("income", 0.0, 1_000_000_000.0, "Declared annual income."),
    FeatureSpec("loan_to_income_ratio", 0.0, 100.0, "Requested principal divided by income."),
    FeatureSpec("debt_to_income_ratio", 0.0, 10.0, "Declared existing debt-to-income ratio."),
    FeatureSpec("account_age_days", 0.0, 36_500.0, "Age of the platform account."),
    FeatureSpec("bank_account_age_days", 0.0, 36_500.0, "Age of the linked bank account."),
    FeatureSpec(
        "origin_balance_before", 0.0, 1_000_000_000_000.0, "Balance before the source transaction."
    ),
    FeatureSpec("amount_to_balance_ratio", 0.0, 100.0, "Request divided by prior balance."),
    FeatureSpec("balance_change_ratio", -100.0, 100.0, "Normalized source balance change."),
    FeatureSpec(
        "applications_last_hour", 0.0, 1_000_000.0, "Prior applications by this user in one hour."
    ),
    FeatureSpec(
        "applications_last_day", 0.0, 1_000_000.0, "Prior applications by this user in one day."
    ),
    FeatureSpec(
        "transaction_frequency_24h", 0.0, 1_000_000.0, "Prior transactions by this user in one day."
    ),
    FeatureSpec(
        "transaction_amount_deviation", -20.0, 20.0, "Winsorized z-score against prior amounts."
    ),
    FeatureSpec("device_changes_30d", 0.0, 1_000_000.0, "Prior device changes in thirty days."),
    FeatureSpec("login_frequency_24h", 0.0, 1_000_000.0, "Prior logins in one day."),
    FeatureSpec("failed_login_attempts_24h", 0.0, 1_000_000.0, "Prior failed logins in one day."),
    FeatureSpec(
        "previous_rejected_applications", 0.0, 1_000_000.0, "Earlier rejected applications."
    ),
    FeatureSpec("is_new_device", 0.0, 1.0, "Device has not previously been observed for the user."),
    FeatureSpec(
        "unusual_login_location", 0.0, 1.0, "Coarse location differs from established behavior."
    ),
    FeatureSpec(
        "shared_device_user_count", 0.0, 1_000_000.0, "Prior users associated with this device."
    ),
    FeatureSpec("shared_ip_user_count", 0.0, 1_000_000.0, "Prior users associated with this IP."),
    FeatureSpec(
        "shared_bank_user_count", 0.0, 1_000_000.0, "Prior users associated with this bank account."
    ),
    FeatureSpec("hour_sin", -1.0, 1.0, "Cyclic local application-hour sine."),
    FeatureSpec("hour_cos", -1.0, 1.0, "Cyclic local application-hour cosine."),
    FeatureSpec("hour_of_day_deviation", 0.0, 12.0, "Circular hours from the user's usual hour."),
    FeatureSpec(
        "is_night_application", 0.0, 1.0, "Application submitted from midnight through 05:59."
    ),
)

FEATURE_NAMES: Final[tuple[str, ...]] = tuple(spec.name for spec in FEATURE_SPECS)


def feature_schema_manifest() -> dict[str, object]:
    """Return JSON-serializable metadata persisted beside model artifacts."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "features": [spec.to_dict() for spec in FEATURE_SPECS],
    }


def validate_feature_vector(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a validated float64 vector in canonical order."""

    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != len(FEATURE_SPECS):
        raise ValueError(
            f"expected a one-dimensional vector of {len(FEATURE_SPECS)} features, "
            f"got shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("feature vector contains NaN or infinite values")
    for index, spec in enumerate(FEATURE_SPECS):
        value = float(values[index])
        if not spec.minimum <= value <= spec.maximum:
            raise ValueError(
                f"feature {spec.name!r}={value} is outside [{spec.minimum}, {spec.maximum}]"
            )
    return values


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate exact columns, order, numeric values, and feature bounds."""

    if tuple(frame.columns) != FEATURE_NAMES:
        raise ValueError(
            "feature columns do not match the ordered contract: "
            f"expected {FEATURE_NAMES}, got {tuple(frame.columns)}"
        )
    values = frame.to_numpy(dtype=np.float64, copy=True)
    for row in values:
        validate_feature_vector(row)
    return pd.DataFrame(values, columns=FEATURE_NAMES, index=frame.index)
