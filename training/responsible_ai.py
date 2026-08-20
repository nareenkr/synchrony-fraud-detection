"""Generate a basic non-sensitive segment review for prototype decision support."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.fraud.anomaly import AnomalyScorer
from backend.app.fraud.risk_engine import RiskEngine, load_risk_config
from backend.app.fraud.supervised import load_classifier_bundle
from training.hybrid_eval import _component_frame, _weighted_scores
from training.replay import replay_splits
from training.train_classifier import load_prepared_splits


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def segment_metrics(labels: pd.Series, flagged: pd.Series) -> dict[str, Any]:
    y = labels.astype(int).to_numpy()
    prediction = flagged.astype(bool).to_numpy()
    tp = int(((y == 1) & prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    tn = int(((y == 0) & ~prediction).sum())
    positives = tp + fn
    negatives = tn + fp
    selected = tp + fp
    return {
        "support": int(len(y)),
        "fraud_cases": positives,
        "precision": tp / selected if selected else None,
        "recall": tp / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
        "review_rate": float(prediction.mean()) if len(prediction) else None,
        "recall_95ci": wilson_interval(tp, positives),
        "false_positive_rate_95ci": wilson_interval(fp, negatives),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _bands(frame: pd.DataFrame) -> dict[str, pd.Series]:
    account_age = pd.cut(
        frame["account_age_days"],
        bins=[-1, 7, 30, 180, 365, float("inf")],
        labels=["0-7d", "8-30d", "31-180d", "181-365d", "366d+"],
    )
    amount = pd.qcut(
        frame["requested_loan_amount"], q=3, labels=["lower", "middle", "upper"], duplicates="drop"
    )
    ratio = frame["requested_loan_amount"] / frame["income"].replace(0, np.nan)
    ratio_band = pd.cut(
        ratio,
        bins=[-float("inf"), 0.25, 0.75, 1.5, float("inf")],
        labels=["<=0.25", "0.25-0.75", "0.75-1.5", ">1.5"],
    )
    observed = (
        frame[["income", "account_age_days", "bank_account_age_days", "device_id", "ip_address"]]
        .notna()
        .sum(axis=1)
    )
    completeness = pd.cut(
        observed,
        bins=[-1, 2, 4, 5],
        labels=["limited", "partial", "complete"],
    )
    return {
        "account_tenure_band": account_age.astype(str),
        "requested_amount_tertile": amount.astype(str),
        "loan_to_income_band": ratio_band.astype(str),
        "data_completeness": completeness.astype(str),
    }


def evaluate_segments(
    source: pd.DataFrame,
    labels: pd.Series,
    hybrid_scores: np.ndarray,
    *,
    threshold: float,
    minimum_support: int = 5,
) -> dict[str, Any]:
    flagged = pd.Series(hybrid_scores >= threshold, index=source.index)
    report: dict[str, Any] = {
        "overall": segment_metrics(labels.reset_index(drop=True), flagged.reset_index(drop=True)),
        "segments": {},
        "minimum_support": minimum_support,
    }
    for dimension, groups in _bands(source.reset_index(drop=True)).items():
        values: dict[str, Any] = {}
        for group in sorted(groups.dropna().unique()):
            mask = groups == group
            metrics = segment_metrics(
                labels.reset_index(drop=True)[mask], flagged.reset_index(drop=True)[mask]
            )
            metrics["interpretation_suppressed"] = metrics["support"] < minimum_support
            values[str(group)] = metrics
        report["segments"][dimension] = values
    return report


def render_markdown(report: dict[str, Any]) -> str:
    source_kind = report.get("dataset_source", {}).get("kind", "unknown")
    lines = [
        "# Responsible-AI Segment Review",
        "",
        "This prototype analysis uses non-sensitive operational segments only. It is not a "
        "legal fairness assessment and cannot establish production suitability.",
        "",
        "| Dimension | Segment | Support | Fraud cases | Recall | FPR | Review rate | Note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    def display(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.3f}"

    for dimension, groups in report["segments"].items():
        for name, metrics in groups.items():
            note = (
                "low support; interpretation suppressed"
                if metrics["interpretation_suppressed"]
                else ""
            )
            lines.append(
                f"| {dimension} | {name} | {metrics['support']} | {metrics['fraud_cases']} | "
                f"{display(metrics['recall'])} | {display(metrics['false_positive_rate'])} | "
                f"{display(metrics['review_rate'])} | {note} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- Race, religion, gender, and other protected traits are not model inputs and are "
            f"not present in the prepared `{source_kind}` data.",
            "- Geography is omitted from segmentation because it may proxy protected status.",
            "- Synthetic lending attributes and sparse fraud labels make disparities unstable; "
            "confidence intervals and support counts must accompany every rate.",
            "- Label bias, domain shift, concept drift, and selective investigation labels remain "
            "unmeasured risks.",
            "- False positives can delay access to credit. The manual-review band, investigator "
            "context, and an appeal path are required safeguards.",
            "- HIGH_RISK is an investigation priority, not authority for autonomous denial.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--classifier-bundle", type=Path, default=Path("artifacts/supervised-v1"))
    parser.add_argument(
        "--anomaly-artifact", type=Path, default=Path("artifacts/anomaly-v1.joblib")
    )
    parser.add_argument("--risk-config", type=Path, default=Path("config/risk.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/responsible_ai.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/responsible_ai.md"))
    parser.add_argument("--minimum-support", type=int, default=5)
    args = parser.parse_args(argv)

    splits = load_prepared_splits(args.data_dir)
    matrices = replay_splits(splits)
    classifier = load_classifier_bundle(args.classifier_bundle)
    anomaly = AnomalyScorer.load(args.anomaly_artifact)
    config, version = load_risk_config(args.risk_config)
    engine = RiskEngine(config, version)
    components = _component_frame(matrices.test, classifier, anomaly)
    hybrid_scores = _weighted_scores(components, engine.normalized_weights)
    report = evaluate_segments(
        splits.test,
        matrices.test.labels,
        hybrid_scores,
        threshold=config.thresholds.manual_review / 100.0,
        minimum_support=args.minimum_support,
    )
    prepared_manifest_path = args.data_dir / "manifest.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    source_kind = prepared_manifest.get("source", {}).get("kind", "unknown")
    report.update(
        {
            "scope": f"untouched chronological {source_kind} test partition",
            "dataset_source": prepared_manifest.get("source", {}),
            "sensitive_features_used": [],
            "geography_segmented": False,
            "prototype_decision_support_only": True,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.output), "markdown": str(args.markdown)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
