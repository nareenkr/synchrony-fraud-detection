from __future__ import annotations

import pandas as pd
import pytest

from backend.app.features import FEATURE_NAMES
from backend.app.fraud.graph import GraphConfig, GraphScorer


def feature_row(**updates: float) -> pd.DataFrame:
    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    values["hour_cos"] = 1.0
    values.update(updates)
    return pd.DataFrame([values], columns=FEATURE_NAMES)


def test_canonical_normal_suspicious_ring_graph_direction() -> None:
    scorer = GraphScorer()
    normal = scorer.score(feature_row())
    suspicious = scorer.score(feature_row(shared_device_user_count=1.0, shared_ip_user_count=2.0))
    ring = scorer.score(
        feature_row(
            shared_device_user_count=7.0,
            shared_ip_user_count=9.0,
            shared_bank_user_count=4.0,
        )
    )

    assert normal.risk == 0
    assert normal.signals == ()
    assert normal.risk < suspicious.risk < ring.risk <= 1
    assert {signal.code for signal in suspicious.signals} == {"SHARED_DEVICE", "SHARED_IP"}
    assert {signal.code for signal in ring.signals} == {
        "SHARED_DEVICE",
        "SHARED_IP",
        "SHARED_BANK_ACCOUNT",
    }
    assert all(signal.source == "graph" for signal in ring.signals)


def test_graph_mapping_input_is_deterministic_and_signals_are_safe() -> None:
    row = feature_row(shared_bank_user_count=2.0).iloc[0]
    first = GraphScorer().score(row)
    second = GraphScorer().score(row.to_dict())

    assert first == second
    assert 0 < first.risk < 1
    assert len(first.signals) == 1
    assert "applicant identities" in first.signals[0].message


def test_graph_rejects_invalid_input_and_config() -> None:
    with pytest.raises(ValueError, match="ordered feature contract"):
        GraphScorer().score(feature_row().loc[:, reversed(FEATURE_NAMES)])
    with pytest.raises(ValueError, match="strictly ordered"):
        GraphConfig(ip_low=2, ip_high=1)
