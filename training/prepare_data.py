"""Deterministic, provenance-aware preparation of PaySim-shaped data.

This module deliberately performs no network access. Pass a local PaySim CSV to
the CLI, or omit it to create a small development fixture. Enrichment happens
before a chronological split, but every derived behavioral value uses prior rows
only. Any stateful estimator supplied to :func:`fit_transform_train_only` is fit
exactly once and only on the training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.schemas import LoanApplicationEvent

DEFAULT_SEED = 20260819
SOURCE_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)
DERIVED_COLUMNS = (
    "applications_last_hour",
    "applications_last_day",
    "transaction_frequency",
    "transaction_amount_deviation",
)
SYNTHETIC_COLUMNS = (
    "account_age_days",
    "device_id",
    "ip_address",
    "geographic_region",
    "device_changes_30d",
    "login_frequency_24h",
    "failed_login_attempts_24h",
    "unusual_login_location",
    "income",
    "requested_loan_amount",
    "debt_to_income_ratio",
    "bank_account_id",
    "bank_account_age_days",
    "previous_rejected_applications",
    "shared_ip_user_count",
    "shared_device_user_count",
    "hour_of_day_deviation",
)


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def row_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


@dataclass(frozen=True)
class PreparedDataset:
    splits: DatasetSplits
    manifest: dict[str, Any]


def _digest(seed: int, namespace: str, key: str) -> bytes:
    return hashlib.sha256(f"{seed}|{namespace}|{key}".encode()).digest()


def _integer(seed: int, namespace: str, key: str, low: int, high: int) -> int:
    """Stable integer in the inclusive range, independent of row ordering."""
    width = high - low + 1
    return low + int.from_bytes(_digest(seed, namespace, key)[:8], "big") % width


def _pseudonym(seed: int, namespace: str, raw: str, length: int = 16) -> str:
    return f"{namespace}-{_digest(seed, namespace, raw).hex()[:length]}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(frame: pd.DataFrame) -> None:
    missing = sorted(set(SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"PaySim input is missing required source columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("PaySim input must contain at least one row")
    numeric = ("step", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud")
    for column in numeric:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            raise ValueError(f"source column {column!r} contains non-numeric or missing values")
    if (pd.to_numeric(frame["step"]) < 0).any() or (pd.to_numeric(frame["amount"]) < 0).any():
        raise ValueError("step and amount must be non-negative")
    if not set(pd.to_numeric(frame["isFraud"]).astype(int).unique()).issubset({0, 1}):
        raise ValueError("isFraud must be binary")


def enrich_paysim(
    source: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    epoch: datetime = datetime(2024, 1, 1, tzinfo=UTC),
) -> pd.DataFrame:
    """Return deterministic lending enrichment with point-in-time derivations.

    Synthetic values are keyed by source entity or stable source-row identity;
    they do not use the fraud label. Reordering the input therefore does not
    change generated values for otherwise identical rows.
    """
    _validate_source(source)
    if epoch.tzinfo is None or epoch.utcoffset() is None:
        raise ValueError("epoch must be timezone-aware")

    data = source.copy(deep=True)
    data["_source_order"] = range(len(data))
    data["step"] = pd.to_numeric(data["step"]).astype(int)
    data["amount"] = pd.to_numeric(data["amount"]).astype(float)
    data = data.sort_values(["step", "_source_order"], kind="stable").reset_index(drop=True)
    data["event_timestamp"] = pd.to_datetime(
        [epoch + timedelta(hours=int(step)) for step in data["step"]], utc=True
    )

    generated: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        raw_user = str(row.nameOrig)
        stable_row = f"{row.step}|{row.nameOrig}|{row.nameDest}|{float(row.amount):.6f}"
        user_id = _pseudonym(seed, "usr", raw_user)
        income = float(_integer(seed, "income", raw_user, 24_000, 240_000))
        requested = max(100.0, min(float(row.amount), 10_000_000.0))
        device_bucket = _integer(seed, "device", raw_user, 0, 9_999)
        ip_bucket = _integer(seed, "ip", raw_user, 1, 250)
        bank_id = _pseudonym(seed, "bank", str(row.nameDest))
        generated.append(
            {
                "application_id": _pseudonym(seed, "app", stable_row, 20),
                "user_id": user_id,
                "fraud_label": int(row.isFraud),
                "account_age_days": _integer(seed, "account-age", raw_user, 0, 3_650),
                "device_id": f"dev-{device_bucket:04d}",
                "ip_address": (
                    f"10.{ip_bucket // 250}.{ip_bucket % 250}."
                    f"{_integer(seed, 'ip-host', raw_user, 1, 254)}"
                ),
                "geographic_region": ("NORTH", "SOUTH", "EAST", "WEST")[
                    _integer(seed, "region", raw_user, 0, 3)
                ],
                "device_changes_30d": _integer(seed, "device-changes", raw_user, 0, 6),
                "login_frequency_24h": _integer(seed, "login-frequency", raw_user, 0, 30),
                "failed_login_attempts_24h": _integer(seed, "failed-logins", stable_row, 0, 5),
                "unusual_login_location": bool(
                    _integer(seed, "unusual-location", stable_row, 0, 19) == 0
                ),
                "income": income,
                "requested_loan_amount": requested,
                "debt_to_income_ratio": min(requested / income, 10.0),
                "bank_account_id": bank_id,
                "bank_account_age_days": _integer(seed, "bank-age", str(row.nameDest), 0, 5_000),
                "previous_rejected_applications": _integer(seed, "rejections", raw_user, 0, 4),
                "hour_of_day_deviation": float(abs((int(row.step) % 24) - 14) / 12),
            }
        )
    data = pd.concat([data, pd.DataFrame(generated)], axis=1)

    # Strictly point-in-time: snapshot counts before recording the current row.
    user_hour: dict[str, deque[datetime]] = defaultdict(deque)
    user_day: dict[str, deque[datetime]] = defaultdict(deque)
    user_amounts: dict[str, list[float]] = defaultdict(list)
    device_users: dict[str, set[str]] = defaultdict(set)
    ip_users: dict[str, set[str]] = defaultdict(set)
    derived: list[dict[str, float | int]] = []
    for row in data.itertuples(index=False):
        now = row.event_timestamp.to_pydatetime()
        user = row.user_id
        while user_hour[user] and user_hour[user][0] < now - timedelta(hours=1):
            user_hour[user].popleft()
        while user_day[user] and user_day[user][0] < now - timedelta(days=1):
            user_day[user].popleft()
        history = user_amounts[user]
        baseline = sum(history) / len(history) if history else float(row.amount)
        deviation = abs(float(row.amount) - baseline) / max(abs(baseline), 1.0)
        derived.append(
            {
                "applications_last_hour": len(user_hour[user]),
                "applications_last_day": len(user_day[user]),
                "transaction_frequency": len(history),
                "transaction_amount_deviation": deviation,
                "shared_ip_user_count": len(ip_users[row.ip_address]),
                "shared_device_user_count": len(device_users[row.device_id]),
            }
        )
        user_hour[user].append(now)
        user_day[user].append(now)
        history.append(float(row.amount))
        device_users[row.device_id].add(user)
        ip_users[row.ip_address].add(user)
    data = pd.concat([data, pd.DataFrame(derived)], axis=1)
    return data.drop(columns=["_source_order"])


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    timestamp_column: str = "event_timestamp",
) -> DatasetSplits:
    """Split chronologically without allowing equal timestamps across borders."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than one")
    if timestamp_column not in frame:
        raise ValueError(f"missing timestamp column {timestamp_column!r}")
    ordered = frame.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    unique_times = list(pd.unique(ordered[timestamp_column]))
    if len(unique_times) < 3:
        raise ValueError(
            "at least three distinct timestamps are required for chronological splitting"
        )
    train_groups = max(1, min(len(unique_times) - 2, int(len(unique_times) * train_fraction)))
    validation_groups = max(1, int(len(unique_times) * validation_fraction))
    validation_groups = min(validation_groups, len(unique_times) - train_groups - 1)
    train_end = unique_times[train_groups - 1]
    validation_end = unique_times[train_groups + validation_groups - 1]
    train = ordered.loc[ordered[timestamp_column] <= train_end].reset_index(drop=True)
    validation = ordered.loc[
        (ordered[timestamp_column] > train_end) & (ordered[timestamp_column] <= validation_end)
    ].reset_index(drop=True)
    test = ordered.loc[ordered[timestamp_column] > validation_end].reset_index(drop=True)
    return DatasetSplits(train=train, validation=validation, test=test)


def fit_transform_train_only(
    transformer: Any,
    splits: DatasetSplits,
    feature_columns: Sequence[str],
) -> DatasetSplits:
    """Fit a preprocessing transformer once on train, then transform each split."""
    transformer.fit(splits.train.loc[:, feature_columns])
    transformed = []
    for partition in (splits.train, splits.validation, splits.test):
        values = transformer.transform(partition.loc[:, feature_columns])
        transformed.append(pd.DataFrame(values, index=partition.index))
    return DatasetSplits(*transformed)


def prepare_dataset(
    source: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    source_metadata: Mapping[str, Any] | None = None,
) -> PreparedDataset:
    enriched = enrich_paysim(source, seed=seed)
    splits = chronological_split(
        enriched,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    content_hash = hashlib.sha256(
        enriched.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    entity_overlap: dict[str, dict[str, int]] = {}
    for column in ("user_id", "device_id", "ip_address", "bank_account_id"):
        train_entities = set(splits.train[column].dropna().astype(str))
        validation_entities = set(splits.validation[column].dropna().astype(str))
        test_entities = set(splits.test[column].dropna().astype(str))
        entity_overlap[column] = {
            "train_validation": len(train_entities & validation_entities),
            "train_test": len(train_entities & test_entities),
            "validation_test": len(validation_entities & test_entities),
        }
    manifest = {
        "schema_version": 1,
        "source": dict(source_metadata or {"kind": "in_memory_paysim_shaped"}),
        "seed": seed,
        "row_count": len(enriched),
        "split_row_counts": splits.row_counts(),
        "content_sha256": content_hash,
        "split_strategy": "chronological_distinct_timestamp_groups",
        "fit_policy": "fit preprocessors on train only; validation and test are transform-only",
        "label_conditioning": False,
        "entity_overlap_audit": entity_overlap,
        "identifier_feature_policy": (
            "identifiers are excluded from model inputs; overlaps are reported for audit only"
        ),
        "provenance": {
            "source": list(SOURCE_COLUMNS),
            "derived_from_source": list(DERIVED_COLUMNS),
            "synthetic": list(SYNTHETIC_COLUMNS),
        },
    }
    return PreparedDataset(splits=splits, manifest=manifest)


def demo_scenarios(
    base_time: datetime | None = None,
) -> Mapping[str, tuple[LoanApplicationEvent, ...]]:
    """Canonical deterministic sequences for the dashboard demo and tests."""
    start = base_time or datetime(2026, 8, 19, 10, 0, tzinfo=UTC)
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("base_time must be timezone-aware")

    normal = LoanApplicationEvent(
        application_id="APP-NORMAL-001",
        user_id="USR-NORMAL",
        event_timestamp=start,
        requested_loan_amount=5_000.0,
        income=80_000.0,
        debt_to_income_ratio=0.18,
        account_age_days=1_400,
        bank_account_age_days=1_800,
        device_id="DEV-NORMAL",
        ip_address="198.51.100.10",
        bank_account_id="BANK-NORMAL",
        geographic_region="WEST",
        device_changes_30d=0,
        login_frequency_24h=3,
        failed_login_attempts_24h=0,
        previous_rejected_applications=0,
        unusual_login_location=False,
        transaction_amount=200.0,
        transaction_frequency_24h=4,
        transaction_amount_deviation=0.1,
        origin_balance_before=50_000.0,
        origin_balance_after=45_000.0,
    )
    suspicious = tuple(
        LoanApplicationEvent(
            application_id=f"APP-SUSPICIOUS-{index + 1:03d}",
            user_id="USR-SUSPICIOUS",
            event_timestamp=start + timedelta(minutes=5 * index),
            requested_loan_amount=45_000.0,
            income=52_000.0,
            debt_to_income_ratio=0.86,
            account_age_days=2,
            bank_account_age_days=5,
            device_id="DEV-SUSPICIOUS-NEW",
            ip_address="203.0.113.80",
            bank_account_id="BANK-SUSPICIOUS",
            geographic_region="EAST",
            device_changes_30d=4,
            login_frequency_24h=25,
            failed_login_attempts_24h=5,
            previous_rejected_applications=2,
            unusual_login_location=True,
            transaction_amount=18_000.0,
            transaction_frequency_24h=30,
            transaction_amount_deviation=4.0,
        )
        for index in range(4)
    )
    ring = tuple(
        LoanApplicationEvent(
            application_id=f"APP-RING-{index + 1:03d}",
            user_id=f"USR-RING-{index + 1:03d}",
            event_timestamp=start + timedelta(seconds=30 * index),
            requested_loan_amount=90_000.0,
            income=35_000.0,
            debt_to_income_ratio=2.57,
            account_age_days=index,
            bank_account_age_days=index + 1,
            device_id="DEV-RING-SHARED",
            ip_address="192.0.2.66",
            bank_account_id=f"BANK-RING-{index + 1:03d}",
            geographic_region="NORTH",
            device_changes_30d=9,
            login_frequency_24h=80,
            failed_login_attempts_24h=12,
            previous_rejected_applications=5,
            unusual_login_location=True,
            transaction_amount=75_000.0,
            transaction_frequency_24h=100,
            transaction_amount_deviation=12.0,
        )
        for index in range(6)
    )
    return {"normal": (normal,), "suspicious": suspicious, "fraud_ring": ring}


def _development_source(rows: int = 600) -> pd.DataFrame:
    """Create a small PaySim-shaped source fixture for executable local checks.

    Fraud labels describe source-level transfer behavior that drains the origin
    balance. Lending, device, network, login, and graph enrichment remains
    independent of this label, as verified by the label-flip regression test.
    The fixture intentionally has more fraud support than production prevalence
    so every chronological split can exercise metric and threshold code.
    """
    records = []
    for index in range(rows):
        amount = float(250 + (index * 7919) % 75_000)
        fraud = int(index % 47 == 0)
        origin_before = amount if fraud else amount * 2
        origin_after = 0.0 if fraud else amount
        records.append(
            {
                "step": index,
                "type": "TRANSFER" if index % 3 else "CASH_OUT",
                "amount": amount,
                "nameOrig": f"C{index % 61:05d}",
                "oldbalanceOrg": origin_before,
                "newbalanceOrig": origin_after,
                "nameDest": f"M{index % 83:05d}",
                "oldbalanceDest": amount / 2,
                "newbalanceDest": amount * 1.5,
                "isFraud": fraud,
                "isFlaggedFraud": int(fraud and amount > 50_000),
            }
        )
    return pd.DataFrame.from_records(records, columns=SOURCE_COLUMNS)


def _write_prepared(prepared: PreparedDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("train", prepared.splits.train),
        ("validation", prepared.splits.validation),
        ("test", prepared.splits.test),
    ):
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(prepared.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local PaySim CSV (no download is performed)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.input:
        source = pd.read_csv(args.input)
        source_metadata = {
            "kind": "public_paysim_csv",
            "filename": args.input.name,
            "input_sha256": _file_sha256(args.input),
        }
    else:
        source = _development_source()
        source_metadata = {
            "kind": "development_fixture",
            "generator": "training.prepare_data._development_source",
            "oversampled_for_pipeline_verification": True,
        }
    prepared = prepare_dataset(source, seed=args.seed, source_metadata=source_metadata)
    _write_prepared(prepared, args.output_dir)
    print(json.dumps(prepared.manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
