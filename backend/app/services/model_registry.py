"""Validated loading of the immutable runtime model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.features import feature_schema_manifest
from backend.app.fraud.anomaly import AnomalyScorer
from backend.app.fraud.risk_engine import RiskEngine
from backend.app.fraud.supervised import SupervisedModel, load_classifier_bundle


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    supervised: SupervisedModel
    anomaly: AnomalyScorer

    @classmethod
    def load(cls, classifier_bundle: str | Path, anomaly_artifact: str | Path) -> ModelRegistry:
        return cls(
            supervised=load_classifier_bundle(classifier_bundle),
            anomaly=AnomalyScorer.load(anomaly_artifact),
        )

    def safe_info(self) -> dict[str, object]:
        manifest = self.supervised.manifest
        metrics = manifest.get("test_evaluation", {})
        safe_metrics = {
            key: metrics.get(key)
            for key in (
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "false_positive_rate",
            )
        }
        return {
            "model_version": self.supervised.model_version,
            "model_name": self.supervised.model_name,
            "feature_schema_version": manifest.get("feature_schema_version"),
            "classifier_threshold": self.supervised.threshold,
            "metrics": safe_metrics,
            "prototype_only": True,
        }


@dataclass(frozen=True, slots=True)
class ModelInfoService:
    registry: ModelRegistry
    risk_engine: RiskEngine

    def safe_info(self) -> dict[str, object]:
        info = self.registry.safe_info()
        info.update(
            {
                "risk_config_version": self.risk_engine.config_version,
                "thresholds": {
                    "manual_review": self.risk_engine.config.thresholds.manual_review,
                    "high_risk": self.risk_engine.config.thresholds.high_risk,
                },
                "weights": dict(self.risk_engine.normalized_weights),
                "features": feature_schema_manifest()["features"],
            }
        )
        return info
