"""Evaluate the frozen hybrid fraud score and component ablations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.fraud.anomaly import AnomalyScorer
from backend.app.fraud.behavioral import BehavioralScorer
from backend.app.fraud.graph import GraphScorer
from backend.app.fraud.risk_engine import RiskEngine, load_risk_config
from backend.app.fraud.supervised import (
    SupervisedModel,
    binary_classification_metrics,
    load_classifier_bundle,
)
from training.replay import ModelMatrix, replay_splits
from training.train_classifier import load_prepared_splits


def _component_frame(
    matrix: ModelMatrix,
    supervised: SupervisedModel,
    anomaly: AnomalyScorer,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    supervised_scores = supervised.predict_proba(matrix.features)
    anomaly_scores = anomaly.score_samples(matrix.features)
    behavioral = BehavioralScorer()
    graph = GraphScorer()
    for index, (_, feature_row) in enumerate(matrix.features.iterrows()):
        rows.append(
            {
                "supervised": float(supervised_scores[index]),
                "anomaly": float(anomaly_scores[index]),
                "behavioral": behavioral.score(feature_row).risk,
                "graph": graph.score(feature_row).risk,
            }
        )
    return pd.DataFrame(rows)


def _weighted_scores(
    components: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    exclude: str | None = None,
) -> np.ndarray:
    active = {name: value for name, value in weights.items() if name != exclude and value > 0}
    total = sum(active.values())
    if total <= 0:
        raise ValueError("ablation removed every positive risk weight")
    result = sum(components[name].to_numpy() * value for name, value in active.items()) / total
    return np.asarray(result, dtype=float)


def evaluate_hybrid(
    matrix: ModelMatrix,
    supervised: SupervisedModel,
    anomaly: AnomalyScorer,
    risk_engine: RiskEngine,
) -> dict[str, Any]:
    components = _component_frame(matrix, supervised, anomaly)
    weights = risk_engine.normalized_weights
    review_threshold = risk_engine.config.thresholds.manual_review / 100.0
    high_threshold = risk_engine.config.thresholds.high_risk / 100.0
    full_scores = _weighted_scores(components, weights)
    metrics = binary_classification_metrics(matrix.labels, full_scores, review_threshold)
    high_risk_metrics = binary_classification_metrics(matrix.labels, full_scores, high_threshold)
    classifier_metrics = binary_classification_metrics(
        matrix.labels, components["supervised"], supervised.threshold
    )
    ablations = {}
    for name in components.columns:
        scores = _weighted_scores(components, weights, exclude=name)
        ablations[f"without_{name}"] = binary_classification_metrics(
            matrix.labels, scores, review_threshold, include_curves=False
        )
    return {
        "evaluation_scope": "untouched chronological test partition",
        "flagged_definition": "MANUAL_REVIEW or HIGH_RISK",
        "review_threshold": review_threshold,
        "high_risk_threshold": high_threshold,
        "component_weights": dict(weights),
        "component_summary": {
            name: {
                "mean": float(components[name].mean()),
                "minimum": float(components[name].min()),
                "maximum": float(components[name].max()),
            }
            for name in components.columns
        },
        "classifier_only": classifier_metrics,
        "hybrid_flagged": metrics,
        "hybrid_high_risk": high_risk_metrics,
        "ablations": ablations,
        "limitations": (
            "Development-fixture metrics are unstable and do not establish production, "
            "fairness, or autonomous-decision readiness."
        ),
    }


def _metric(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Hybrid Fraud System Evaluation",
        "",
        str(report["evaluation_scope"]).capitalize() + ".",
        "",
        "| System | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | Review rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Classifier only", "classifier_only"),
        ("Hybrid flagged", "hybrid_flagged"),
        ("Hybrid high risk", "hybrid_high_risk"),
    ):
        metrics = report[key]
        lines.append(
            f"| {label} | {_metric(metrics['precision'])} | {_metric(metrics['recall'])} | "
            f"{_metric(metrics['f1'])} | {_metric(metrics['roc_auc'])} | "
            f"{_metric(metrics['pr_auc'])} | {_metric(metrics['false_positive_rate'])} | "
            f"{_metric(metrics['review_rate'])} |"
        )
    lines.extend(["", "## Component ablation", ""])
    lines.extend(
        [
            "| Ablation | Recall | FPR | PR-AUC | Review rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in report["ablations"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {_metric(metrics['recall'])} | "
            f"{_metric(metrics['false_positive_rate'])} | {_metric(metrics['pr_auc'])} | "
            f"{_metric(metrics['review_rate'])} |"
        )
    lines.extend(["", "## Limitations", "", str(report["limitations"]), ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--classifier-bundle", type=Path, default=Path("artifacts/supervised-v1"))
    parser.add_argument(
        "--anomaly-artifact", type=Path, default=Path("artifacts/anomaly-v1.joblib")
    )
    parser.add_argument("--risk-config", type=Path, default=Path("config/risk.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/hybrid_evaluation.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/hybrid_evaluation.md"))
    args = parser.parse_args(argv)

    matrices = replay_splits(load_prepared_splits(args.data_dir))
    classifier = load_classifier_bundle(args.classifier_bundle)
    anomaly = AnomalyScorer.load(args.anomaly_artifact)
    config, version = load_risk_config(args.risk_config)
    report = evaluate_hybrid(matrices.test, classifier, anomaly, RiskEngine(config, version))
    prepared_manifest_path = args.data_dir / "manifest.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    source_kind = prepared_manifest.get("source", {}).get("kind", "unknown")
    report["dataset_source"] = prepared_manifest.get("source", {})
    report["limitations"] = (
        f"Metrics use dataset source '{source_kind}' and do not establish production, fairness, "
        "or autonomous-decision readiness. PaySim is simulated transaction data and the lending "
        "context is deterministic synthetic enrichment."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
