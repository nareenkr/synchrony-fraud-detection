from __future__ import annotations

import numpy as np

from backend.app.fraud.anomaly import AnomalyScorer
from backend.app.fraud.risk_engine import RiskEngine, load_risk_config
from backend.app.fraud.supervised import SupervisedModel
from training.hybrid_eval import evaluate_hybrid, render_markdown
from training.prepare_data import _development_source, prepare_dataset
from training.replay import replay_splits


class FixedEstimator:
    def predict_proba(self, frame):  # type: ignore[no-untyped-def]
        probability = np.clip(frame["loan_to_income_ratio"].to_numpy() / 2.0, 0, 1)
        return np.column_stack([1 - probability, probability])


def test_hybrid_report_contains_required_comparison_and_ablations() -> None:
    matrices = replay_splits(prepare_dataset(_development_source(100)).splits)
    anomaly = AnomalyScorer().fit(matrices.train.features)
    classifier = SupervisedModel(
        estimator=FixedEstimator(),
        threshold=0.5,
        model_name="fixed",
        model_version="test",
        manifest={},
    )
    config, version = load_risk_config("config/risk.yaml")
    report = evaluate_hybrid(matrices.test, classifier, anomaly, RiskEngine(config, version))

    assert report["classifier_only"]
    assert report["hybrid_flagged"]
    assert set(report["ablations"]) == {
        "without_supervised",
        "without_anomaly",
        "without_behavioral",
        "without_graph",
    }
    assert "Hybrid Fraud System Evaluation" in render_markdown(report)
