from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.features import FEATURE_NAMES
from backend.app.fraud.behavioral import BehavioralConfig, BehavioralScorer


def feature_row(**updates: float) -> pd.DataFrame:
    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    values.update(
        {
            "income": 80_000.0,
            "account_age_days": 1_200.0,
            "bank_account_age_days": 1_500.0,
            "hour_cos": 1.0,
        }
    )
    values.update(updates)
    return pd.DataFrame([values], columns=FEATURE_NAMES)


def test_canonical_normal_suspicious_ring_direction_and_safe_signals() -> None:
    scorer = BehavioralScorer()
    normal = scorer.score(
        feature_row(
            requested_loan_amount=5_000.0,
            loan_to_income_ratio=0.0625,
            transaction_frequency_24h=4.0,
            transaction_amount_deviation=0.1,
        )
    )
    suspicious = scorer.score(
        feature_row(
            requested_loan_amount=45_000.0,
            income=52_000.0,
            loan_to_income_ratio=0.865,
            account_age_days=2.0,
            applications_last_hour=4.0,
            applications_last_day=7.0,
            transaction_frequency_24h=30.0,
            transaction_amount_deviation=4.0,
            device_changes_30d=4.0,
            failed_login_attempts_24h=5.0,
            is_new_device=1.0,
            unusual_login_location=1.0,
        )
    )
    ring = scorer.score(
        feature_row(
            requested_loan_amount=100_000.0,
            income=35_000.0,
            loan_to_income_ratio=2.86,
            account_age_days=0.0,
            applications_last_hour=12.0,
            applications_last_day=20.0,
            transaction_frequency_24h=100.0,
            transaction_amount_deviation=12.0,
            device_changes_30d=9.0,
            failed_login_attempts_24h=12.0,
            is_new_device=1.0,
            unusual_login_location=1.0,
        )
    )

    assert 0 <= normal.risk < suspicious.risk < ring.risk <= 1
    assert normal.signals == ()
    assert {signal.code for signal in suspicious.signals} >= {
        "APPLICATION_VELOCITY",
        "RECENT_DEVICE_CHANGE",
        "HIGH_LOAN_TO_INCOME",
        "REPEATED_FAILED_LOGINS",
        "UNUSUAL_LOGIN_LOCATION",
        "AMOUNT_DEVIATION",
        "LARGE_LOAN_REQUEST",
    }
    assert all(signal.source == "behavioral" for signal in suspicious.signals)
    assert all(
        "USR" not in signal.message and "DEV" not in signal.message for signal in ring.signals
    )


def test_behavioral_scores_are_deterministic_bounded_and_sorted() -> None:
    scorer = BehavioralScorer()
    row = feature_row(
        requested_loan_amount=40_000.0,
        loan_to_income_ratio=1.0,
        failed_login_attempts_24h=4.0,
        unusual_login_location=1.0,
    )
    first = scorer.score(row)
    second = scorer.score(row.iloc[0])

    assert first == second
    assert np.isfinite(first.risk) and 0 <= first.risk <= 1
    assert [signal.severity for signal in first.signals] == sorted(
        (signal.severity for signal in first.signals), reverse=True
    )


def test_behavioral_rejects_bad_contract_and_configuration() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        BehavioralScorer().score(pd.concat([feature_row(), feature_row()]))
    with pytest.raises(ValueError, match="missing canonical"):
        BehavioralScorer().score({"requested_loan_amount": 1.0})
    with pytest.raises(ValueError, match="strictly ordered"):
        BehavioralConfig(applications_hour_low=5, applications_hour_high=5)
