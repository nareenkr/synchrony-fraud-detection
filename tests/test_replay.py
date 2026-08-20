from __future__ import annotations

from training.prepare_data import _development_source, prepare_dataset
from training.replay import entity_overlap_report, replay_splits


def test_replay_uses_shared_contract_and_preserves_partitions() -> None:
    prepared = prepare_dataset(_development_source(80))
    matrices = replay_splits(prepared.splits)

    assert len(matrices.train.features) == len(prepared.splits.train)
    assert len(matrices.validation.features) == len(prepared.splits.validation)
    assert len(matrices.test.features) == len(prepared.splits.test)
    assert matrices.train.features.isna().sum().sum() == 0
    assert matrices.train.application_ids.is_unique


def test_replay_is_deterministic_and_current_event_is_not_counted() -> None:
    prepared = prepare_dataset(_development_source(80))
    first = replay_splits(prepared.splits)
    second = replay_splits(prepared.splits)

    assert first.train.features.equals(second.train.features)
    assert first.validation.features.equals(second.validation.features)
    assert first.train.features.iloc[0]["applications_last_hour"] == 0


def test_entity_overlap_audit_reports_counts_without_identifiers() -> None:
    prepared = prepare_dataset(_development_source(80))
    report = entity_overlap_report(prepared.splits)

    assert set(report) == {"user_id", "device_id", "ip_address", "bank_account_id"}
    assert all(
        isinstance(count, int) and count >= 0
        for pair_counts in report.values()
        for count in pair_counts.values()
    )
