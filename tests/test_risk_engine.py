from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.app.fraud.risk_engine import RiskConfig, RiskEngine, load_risk_config
from backend.app.schemas import ComponentScores, Decision


class RiskEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config, version = load_risk_config("config/risk.yaml")
        cls.engine = RiskEngine(config, version)

    def test_weighted_score_uses_all_components(self) -> None:
        result = self.engine.combine(
            ComponentScores(
                supervised_probability=0.8,
                anomaly_score=0.6,
                behavioral_risk=0.5,
                graph_risk=0.3,
            )
        )
        expected = 100 * (
            0.8 * self.engine.normalized_weights["supervised"]
            + 0.6 * self.engine.normalized_weights["anomaly"]
            + 0.5 * self.engine.normalized_weights["behavioral"]
            + 0.3 * self.engine.normalized_weights["graph"]
        )
        self.assertEqual(result.risk_score, round(expected, 2))
        self.assertEqual(result.decision, Decision.MANUAL_REVIEW)
        self.assertAlmostEqual(sum(result.normalized_weights.values()), 1.0)

    def test_configured_boundaries_are_inclusive(self) -> None:
        self.assertEqual(self.engine.decision_for(39.99), Decision.APPROVE)
        self.assertEqual(self.engine.decision_for(40), Decision.MANUAL_REVIEW)
        self.assertEqual(self.engine.decision_for(69.99), Decision.MANUAL_REVIEW)
        self.assertEqual(self.engine.decision_for(70), Decision.HIGH_RISK)

    def test_non_unit_weights_are_normalized(self) -> None:
        payload = {
            "schema_version": 1,
            "weights": {"supervised": 2.0, "anomaly": 1.0, "behavioral": 1.0, "graph": 0.0},
            "thresholds": {"manual_review": 30.0, "high_risk": 80.0},
            "labels": {
                "approve": "APPROVE",
                "manual_review": "MANUAL_REVIEW",
                "high_risk": "HIGH_RISK",
            },
        }
        engine = RiskEngine(RiskConfig.model_validate(payload), "test")
        result = engine.combine(
            ComponentScores(
                supervised_probability=1.0,
                anomaly_score=0.0,
                behavioral_risk=0.0,
                graph_risk=0.0,
            )
        )
        self.assertEqual(result.risk_score, 50.0)

    def test_rejects_unordered_thresholds(self) -> None:
        payload = {
            "schema_version": 1,
            "weights": {"supervised": 1.0, "anomaly": 0.0, "behavioral": 0.0, "graph": 0.0},
            "thresholds": {"manual_review": 80.0, "high_risk": 70.0},
            "labels": {
                "approve": "APPROVE",
                "manual_review": "MANUAL_REVIEW",
                "high_risk": "HIGH_RISK",
            },
        }
        with self.assertRaises(ValidationError):
            RiskConfig.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
