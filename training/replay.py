"""Leakage-safe conversion of prepared events into the shared model feature contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from backend.app.features import FEATURE_NAMES, FeatureBuilder
from backend.app.schemas import LoanApplicationEvent
from backend.app.state import MemoryRealtimeStateStore
from training.prepare_data import DatasetSplits


@dataclass(frozen=True, slots=True)
class ModelMatrix:
    features: pd.DataFrame
    labels: pd.Series
    application_ids: pd.Series


@dataclass(frozen=True, slots=True)
class ModelMatrixSplits:
    train: ModelMatrix
    validation: ModelMatrix
    test: ModelMatrix


def _optional(row: pd.Series, name: str, cast: type) -> Any:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    return cast(value)


def _optional_bool(row: pd.Series, name: str) -> bool | None:
    value = row.get(name)
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"prepared field {name!r} must be boolean")


def event_from_prepared_row(row: pd.Series) -> LoanApplicationEvent:
    """Build the strict online event schema from one prepared offline row."""

    timestamp = pd.Timestamp(row["event_timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    occurred_at: datetime = timestamp.to_pydatetime()
    return LoanApplicationEvent(
        application_id=str(row["application_id"]),
        user_id=str(row["user_id"]),
        event_timestamp=occurred_at,
        requested_loan_amount=float(row["requested_loan_amount"]),
        income=_optional(row, "income", float),
        debt_to_income_ratio=_optional(row, "debt_to_income_ratio", float),
        account_age_days=_optional(row, "account_age_days", int),
        bank_account_age_days=_optional(row, "bank_account_age_days", int),
        device_id=_optional(row, "device_id", str),
        ip_address=_optional(row, "ip_address", str),
        bank_account_id=_optional(row, "bank_account_id", str),
        geographic_region=_optional(row, "geographic_region", str),
        device_changes_30d=_optional(row, "device_changes_30d", int),
        login_frequency_24h=_optional(row, "login_frequency_24h", int),
        failed_login_attempts_24h=_optional(row, "failed_login_attempts_24h", int),
        previous_rejected_applications=_optional(row, "previous_rejected_applications", int),
        unusual_login_location=_optional_bool(row, "unusual_login_location"),
        transaction_amount=_optional(row, "amount", float),
        transaction_frequency_24h=_optional(row, "transaction_frequency", int),
        transaction_amount_deviation=_optional(row, "transaction_amount_deviation", float),
        origin_balance_before=_optional(row, "oldbalanceOrg", float),
        origin_balance_after=_optional(row, "newbalanceOrig", float),
    )


def replay_splits(
    splits: DatasetSplits,
    *,
    feature_builder: FeatureBuilder | None = None,
    state_store: MemoryRealtimeStateStore | None = None,
) -> ModelMatrixSplits:
    """Replay all partitions in time order with one prior-only state timeline.

    Validation sees train history and test sees train plus validation history,
    just as later online events would. No feature can observe the current or a
    future event. Partition membership is restored after replay.
    """

    builder = feature_builder or FeatureBuilder()
    state = state_store or MemoryRealtimeStateStore()
    tagged: list[pd.DataFrame] = []
    for partition, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        copy = frame.copy()
        copy["_partition"] = partition
        copy["_partition_order"] = range(len(copy))
        tagged.append(copy)
    timeline = pd.concat(tagged, ignore_index=True)
    timeline["event_timestamp"] = pd.to_datetime(timeline["event_timestamp"], utc=True)
    timeline = timeline.sort_values(
        ["event_timestamp", "_partition", "_partition_order"], kind="stable"
    )

    rows: list[dict[str, Any]] = []
    for _, row in timeline.iterrows():
        event = event_from_prepared_row(row)
        snapshot = state.snapshot_and_record(event)
        feature_row = builder.transform(event, snapshot).iloc[0].to_dict()
        feature_row.update(
            {
                "_partition": row["_partition"],
                "_partition_order": int(row["_partition_order"]),
                "_label": int(row["fraud_label"]),
                "_application_id": event.application_id,
            }
        )
        rows.append(feature_row)

    replayed = pd.DataFrame(rows)

    def extract(partition: str) -> ModelMatrix:
        selected = replayed.loc[replayed["_partition"] == partition].sort_values("_partition_order")
        return ModelMatrix(
            features=selected.loc[:, FEATURE_NAMES].reset_index(drop=True),
            labels=selected["_label"].astype(int).reset_index(drop=True),
            application_ids=selected["_application_id"].astype(str).reset_index(drop=True),
        )

    return ModelMatrixSplits(
        train=extract("train"),
        validation=extract("validation"),
        test=extract("test"),
    )


def entity_overlap_report(splits: DatasetSplits) -> dict[str, dict[str, int]]:
    """Audit pseudonymous entity overlap without exposing identifier values."""

    report: dict[str, dict[str, int]] = {}
    for column in ("user_id", "device_id", "ip_address", "bank_account_id"):
        sets = {
            "train": set(splits.train[column].dropna().astype(str)),
            "validation": set(splits.validation[column].dropna().astype(str)),
            "test": set(splits.test[column].dropna().astype(str)),
        }
        report[column] = {
            "train_validation": len(sets["train"] & sets["validation"]),
            "train_test": len(sets["train"] & sets["test"]),
            "validation_test": len(sets["validation"] & sets["test"]),
        }
    return report
