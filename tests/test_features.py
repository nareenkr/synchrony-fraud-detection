"""Contract tests for shared online/offline feature engineering."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from backend.app.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FeatureBuilder,
    FeatureValidationError,
    validate_feature_frame,
    validate_feature_vector,
)
from backend.app.schemas import LoanApplicationEvent
from backend.app.state import StateSnapshot


class FeatureBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = FeatureBuilder()

    @staticmethod
    def event(**updates: object) -> LoanApplicationEvent:
        values: dict[str, object] = {
            "application_id": "app-1",
            "user_id": "user-1",
            "event_timestamp": datetime(2026, 8, 19, 12, tzinfo=UTC),
            "requested_loan_amount": 10_000.0,
        }
        values.update(updates)
        return LoanApplicationEvent.model_validate(values)

    def test_contract_is_exact_ordered_and_excludes_identifiers_and_label(self) -> None:
        event = self.event(income=50_000.0)
        frame = self.builder.transform(event, {})
        vector = self.builder.transform_vector(event, {})

        self.assertEqual(FeatureBuilder.schema_version, FEATURE_SCHEMA_VERSION)
        self.assertEqual(tuple(frame.columns), FEATURE_NAMES)
        self.assertEqual(frame.shape, (1, len(FEATURE_NAMES)))
        np.testing.assert_allclose(frame.iloc[0].to_numpy(), vector)
        pd.testing.assert_frame_equal(frame, self.builder.transform(event, {}))
        self.assertTrue(np.isfinite(vector).all())
        forbidden = {
            "application_id",
            "user_id",
            "device_id",
            "ip_address",
            "bank_account_id",
            "event_timestamp",
            "fraud_label",
            "is_fraud",
        }
        self.assertTrue(forbidden.isdisjoint(FEATURE_NAMES))

    def test_formulas_cover_financial_behavioral_graph_and_time_features(self) -> None:
        event = self.event(
            event_timestamp=datetime(2026, 8, 19, 3, tzinfo=UTC),
            requested_loan_amount=12_000.0,
            income=60_000.0,
            debt_to_income_ratio=0.35,
            account_age_days=120,
            bank_account_age_days=300,
            transaction_amount=150.0,
            origin_balance_before=1_000.0,
            origin_balance_after=850.0,
            previous_rejected_applications=2,
            unusual_login_location=True,
        )
        snapshot = {
            "applications_last_hour": 3,
            "applications_last_day": 8,
            "transaction_frequency_24h": 14,
            "prior_amount_mean": 100.0,
            "prior_amount_std": 20.0,
            "prior_amount_count": 5,
            "device_changes_30d": 2,
            "login_frequency_24h": 11,
            "failed_login_attempts_24h": 4,
            "last_device_id": "device-old",
            "device_changed": False,
            "shared_device_user_count": 5,
            "shared_ip_user_count": 7,
            "shared_bank_user_count": 2,
            "usual_login_hour": 21.0,
        }

        row = self.builder.transform(event, snapshot).iloc[0]

        self.assertAlmostEqual(row["loan_to_income_ratio"], 0.2)
        self.assertAlmostEqual(row["amount_to_balance_ratio"], 0.15)
        self.assertAlmostEqual(row["balance_change_ratio"], -0.15)
        self.assertAlmostEqual(row["transaction_amount_deviation"], 2.5)
        self.assertEqual(row["applications_last_hour"], 3.0)
        self.assertEqual(row["applications_last_day"], 8.0)
        self.assertEqual(row["shared_device_user_count"], 5.0)
        self.assertEqual(row["shared_ip_user_count"], 7.0)
        self.assertEqual(row["shared_bank_user_count"], 2.0)
        self.assertEqual(row["is_new_device"], 0.0)
        self.assertEqual(row["unusual_login_location"], 1.0)
        self.assertAlmostEqual(row["hour_sin"], np.sqrt(0.5))
        self.assertAlmostEqual(row["hour_cos"], np.sqrt(0.5))
        self.assertEqual(row["hour_of_day_deviation"], 6.0)
        self.assertEqual(row["is_night_application"], 1.0)

    def test_new_user_and_missing_optional_values_are_finite(self) -> None:
        frame = self.builder.transform(self.event(), {})
        row = frame.iloc[0]

        self.assertTrue(np.isfinite(frame.to_numpy()).all())
        self.assertEqual(row["income"], 0.0)
        self.assertEqual(row["loan_to_income_ratio"], 100.0)
        self.assertEqual(row["account_age_days"], 0.0)
        self.assertEqual(row["origin_balance_before"], 0.0)
        self.assertEqual(row["transaction_amount_deviation"], 0.0)
        self.assertEqual(row["applications_last_hour"], 0.0)
        self.assertEqual(row["is_new_device"], 0.0)

        identified_new_user = self.builder.transform(
            self.event(application_id="app-2", device_id="device-new"), {}
        )
        self.assertEqual(identified_new_user.iloc[0]["is_new_device"], 1.0)

    def test_canonical_state_and_supplied_observations_use_max_merge_policy(self) -> None:
        event = self.event(
            device_id="device-current",
            device_changes_30d=7,
            login_frequency_24h=3,
            failed_login_attempts_24h=2,
            transaction_frequency_24h=9,
        )
        snapshot = StateSnapshot(
            as_of=event.event_timestamp,
            applications_last_hour=2,
            applications_last_day=4,
            login_frequency_24h=8,
            failed_login_attempts_24h=1,
            device_changes_30d=5,
            shared_device_user_count=3,
            shared_ip_user_count=4,
            shared_bank_user_count=2,
            last_device_id="device-old",
            device_changed=True,
        )

        row = self.builder.transform(event, snapshot).iloc[0]

        self.assertEqual(row["applications_last_hour"], 2.0)
        self.assertEqual(row["applications_last_day"], 4.0)
        self.assertEqual(row["transaction_frequency_24h"], 9.0)
        self.assertEqual(row["device_changes_30d"], 7.0)
        self.assertEqual(row["login_frequency_24h"], 8.0)
        self.assertEqual(row["failed_login_attempts_24h"], 2.0)
        self.assertEqual(row["is_new_device"], 1.0)

    def test_extreme_valid_values_are_saturated_to_declared_ranges(self) -> None:
        event = self.event(
            requested_loan_amount=10_000_000.0,
            income=1.0,
            device_id="device-current",
            transaction_amount=100_000_000.0,
            transaction_amount_deviation=1_000.0,
            origin_balance_before=0.0,
            origin_balance_after=1_000_000_000.0,
        )
        snapshot = {
            "applications_last_hour": 10**9,
            "applications_last_day": 10**9,
            "last_device_id": "device-1",
            "device_changed": True,
            "shared_device_user_count": 10**9,
            "shared_ip_user_count": 10**9,
            "shared_bank_user_count": 10**9,
        }

        row = self.builder.transform(event, snapshot).iloc[0]

        self.assertEqual(row["loan_to_income_ratio"], 100.0)
        self.assertEqual(row["amount_to_balance_ratio"], 100.0)
        self.assertEqual(row["balance_change_ratio"], 100.0)
        self.assertEqual(row["transaction_amount_deviation"], 20.0)
        self.assertEqual(row["applications_last_hour"], 1_000_000.0)
        self.assertEqual(row["shared_device_user_count"], 1_000_000.0)
        self.assertEqual(row["is_new_device"], 1.0)

    def test_invalid_state_numbers_fail_instead_of_entering_model(self) -> None:
        with self.assertRaisesRegex(FeatureValidationError, "cannot be negative"):
            self.builder.transform(self.event(), {"applications_last_hour": -1})
        with self.assertRaisesRegex(FeatureValidationError, "must be finite"):
            self.builder.transform(self.event(), {"shared_ip_user_count": float("nan")})

    def test_contract_validator_rejects_order_shape_and_nonfinite_values(self) -> None:
        valid = self.builder.transform(self.event(income=40_000.0), {})
        with self.assertRaisesRegex(ValueError, "ordered contract"):
            validate_feature_frame(valid.loc[:, reversed(FEATURE_NAMES)])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            validate_feature_vector(np.zeros((1, len(FEATURE_NAMES))))
        invalid = valid.copy()
        invalid.loc[0, "income"] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            validate_feature_frame(invalid)


if __name__ == "__main__":
    unittest.main()
