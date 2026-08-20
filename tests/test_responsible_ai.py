from __future__ import annotations

import pandas as pd

from training.responsible_ai import evaluate_segments, segment_metrics, wilson_interval


def test_wilson_interval_is_bounded_and_missing_without_support() -> None:
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(5, 10) or (0, 0)
    assert 0 <= low <= 0.5 <= high <= 1


def test_segment_metrics_report_false_positive_and_review_rates() -> None:
    metrics = segment_metrics(pd.Series([0, 0, 1, 1]), pd.Series([False, True, True, False]))
    assert metrics["recall"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["review_rate"] == 0.5


def test_low_support_segment_interpretation_is_suppressed() -> None:
    frame = pd.DataFrame(
        {
            "account_age_days": [1, 20, 200, 500],
            "requested_loan_amount": [100, 200, 300, 400],
            "income": [1_000, 1_000, 1_000, 1_000],
            "bank_account_age_days": [1, 2, 3, 4],
            "device_id": ["a", "b", "c", "d"],
            "ip_address": ["1.1.1.1", "1.1.1.2", "1.1.1.3", "1.1.1.4"],
        }
    )
    report = evaluate_segments(
        frame,
        pd.Series([0, 0, 1, 1]),
        pd.Series([0.1, 0.6, 0.7, 0.2]).to_numpy(),
        threshold=0.4,
        minimum_support=5,
    )
    assert all(
        group["interpretation_suppressed"]
        for groups in report["segments"].values()
        for group in groups.values()
    )
