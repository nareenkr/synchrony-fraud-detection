from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from training.prepare_data import (
    DERIVED_COLUMNS,
    SOURCE_COLUMNS,
    SYNTHETIC_COLUMNS,
    DatasetSplits,
    _development_source,
    chronological_split,
    demo_scenarios,
    enrich_paysim,
    fit_transform_train_only,
    main,
    prepare_dataset,
)


def test_enrichment_is_deterministic_and_provenance_columns_are_separate() -> None:
    source = _development_source(40)
    first = enrich_paysim(source, seed=42)
    second = enrich_paysim(source, seed=42)
    pd.testing.assert_frame_equal(first, second)

    assert set(SOURCE_COLUMNS).issubset(first.columns)
    assert set(DERIVED_COLUMNS).issubset(first.columns)
    assert set(SYNTHETIC_COLUMNS).issubset(first.columns)
    assert set(SOURCE_COLUMNS).isdisjoint(SYNTHETIC_COLUMNS)
    assert first["application_id"].is_unique


def test_synthetic_generation_does_not_condition_on_fraud_label() -> None:
    source = _development_source(30)
    relabeled = source.copy()
    relabeled["isFraud"] = 1 - relabeled["isFraud"]
    original = enrich_paysim(source, seed=7)
    changed = enrich_paysim(relabeled, seed=7)

    pd.testing.assert_frame_equal(
        original.loc[:, list(SYNTHETIC_COLUMNS)],
        changed.loc[:, list(SYNTHETIC_COLUMNS)],
    )
    assert not original["fraud_label"].equals(changed["fraud_label"])


def test_point_in_time_counts_exclude_the_current_row() -> None:
    source = _development_source(4)
    source.loc[:, "nameOrig"] = "SAME-USER"
    source.loc[:, "step"] = [0, 0, 1, 25]
    enriched = enrich_paysim(source)

    assert enriched["applications_last_hour"].tolist() == [0, 1, 2, 0]
    # The event exactly 24 hours earlier remains in the inclusive window;
    # older events have expired and the current event is still excluded.
    assert enriched["applications_last_day"].tolist() == [0, 1, 2, 1]


def test_chronological_split_keeps_timestamp_groups_disjoint() -> None:
    source = _development_source(60)
    source["step"] = np.repeat(np.arange(20), 3)
    enriched = enrich_paysim(source)
    splits = chronological_split(enriched)

    assert splits.train["event_timestamp"].max() < splits.validation["event_timestamp"].min()
    assert splits.validation["event_timestamp"].max() < splits.test["event_timestamp"].min()
    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(enriched)


def test_transformer_fits_train_once_and_only_transforms_other_splits() -> None:
    class SpyTransformer:
        def __init__(self) -> None:
            self.fit_values: pd.DataFrame | None = None
            self.transform_calls = 0

        def fit(self, values: pd.DataFrame) -> SpyTransformer:
            self.fit_values = values.copy()
            return self

        def transform(self, values: pd.DataFrame) -> np.ndarray:
            self.transform_calls += 1
            return values.to_numpy(dtype=float)

    splits = DatasetSplits(
        train=pd.DataFrame({"x": [1.0, 2.0]}),
        validation=pd.DataFrame({"x": [100.0]}),
        test=pd.DataFrame({"x": [200.0]}),
    )
    spy = SpyTransformer()
    transformed = fit_transform_train_only(spy, splits, ["x"])

    assert spy.fit_values is not None
    assert spy.fit_values["x"].tolist() == [1.0, 2.0]
    assert spy.transform_calls == 3
    assert transformed.validation.iloc[0, 0] == 100.0


def test_manifest_is_reproducible_and_discloses_label_policy() -> None:
    source = _development_source(80)
    first = prepare_dataset(source, seed=20260819)
    second = prepare_dataset(source, seed=20260819)

    assert first.manifest == second.manifest
    assert first.manifest["content_sha256"]
    assert first.manifest["source"]["kind"] == "in_memory_paysim_shaped"
    assert first.manifest["label_conditioning"] is False
    assert first.manifest["split_row_counts"] == first.splits.row_counts()
    assert set(first.manifest["entity_overlap_audit"]) == {
        "user_id",
        "device_id",
        "ip_address",
        "bank_account_id",
    }


def test_demo_scenarios_are_deterministic_and_cover_required_cases() -> None:
    base = datetime(2026, 8, 19, 10, tzinfo=UTC)
    first = demo_scenarios(base)
    second = demo_scenarios(base)

    assert first == second
    assert set(first) == {"normal", "suspicious", "fraud_ring"}
    assert len(first["normal"]) == 1
    assert len(first["suspicious"]) >= 3
    assert len({event.user_id for event in first["fraud_ring"]}) >= 3
    assert len({event.device_id for event in first["fraud_ring"]}) == 1
    assert len({event.ip_address for event in first["fraud_ring"]}) == 1


def test_invalid_source_and_naive_demo_time_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing required"):
        enrich_paysim(pd.DataFrame({"step": [0]}))
    with pytest.raises(ValueError, match="timezone-aware"):
        demo_scenarios(datetime(2026, 8, 19, 10))


def test_cli_records_public_input_filename_and_hash(tmp_path) -> None:
    input_path = tmp_path / "paysim.csv"
    output_path = tmp_path / "prepared"
    _development_source(60).to_csv(input_path, index=False)

    assert main(["--input", str(input_path), "--output-dir", str(output_path)]) == 0

    manifest = pd.read_json(output_path / "manifest.json", typ="series")
    assert manifest["source"]["kind"] == "public_paysim_csv"
    assert manifest["source"]["filename"] == "paysim.csv"
    assert len(manifest["source"]["input_sha256"]) == 64
