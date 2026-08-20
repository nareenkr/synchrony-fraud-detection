"""Evaluate a frozen classifier on the untouched test split and write a report.

Example:
    python -m training.evaluate --bundle-dir artifacts/supervised-v1
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from backend.app.fraud.supervised import (
    binary_classification_metrics,
    load_classifier_bundle,
    sha256_file,
)
from training.replay import replay_splits
from training.train_classifier import load_prepared_splits


def evaluate_bundle(bundle_dir: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """Recompute test metrics using the bundle's already-frozen threshold."""

    model = load_classifier_bundle(bundle_dir)
    data_root = Path(data_dir)
    expected_manifest_hash = (
        model.manifest.get("dataset", {}).get("manifest_sha256")
        if isinstance(model.manifest.get("dataset"), dict)
        else None
    )
    prepared_manifest_path = data_root / "manifest.json"
    if expected_manifest_hash is not None:
        if not prepared_manifest_path.is_file():
            raise ValueError("bundle expects a prepared-data manifest, but none was found")
        if sha256_file(prepared_manifest_path) != expected_manifest_hash:
            raise ValueError("prepared-data manifest does not match the training bundle")
    matrices = replay_splits(load_prepared_splits(data_root))
    probabilities = model.predict_proba(matrices.test.features)
    metrics = binary_classification_metrics(
        matrices.test.labels, probabilities, model.threshold, include_curves=True
    )
    return {
        "model_version": model.model_version,
        "model_name": model.model_name,
        "feature_schema_version": model.manifest["feature_schema_version"],
        "threshold_source": "frozen_validation_selection",
        "test_evaluation": metrics,
    }


def _display(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric_table(metrics: dict[str, Any]) -> list[str]:
    rows = ["| Metric | Value |", "|---|---:|"]
    labels = (
        ("Rows", "row_count"),
        ("Fraud cases", "positive_count"),
        ("Non-fraud cases", "negative_count"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("False-positive rate", "false_positive_rate"),
        ("Review rate", "review_rate"),
    )
    rows.extend(f"| {label} | {_display(metrics[key])} |" for label, key in labels)
    return rows


def _curve_table(curve: dict[str, list[float]] | None, columns: tuple[str, ...]) -> list[str]:
    if curve is None:
        return ["Curve is undefined because this test partition contains only one label class."]
    rows = [f"| {' | '.join(columns)} |", f"|{'|'.join(['---:'] * len(columns))}|"]
    length = max(len(curve[column]) for column in columns)
    for index in range(length):
        cells = [
            _display(curve[column][index]) if index < len(curve[column]) else "N/A"
            for column in columns
        ]
        rows.append(f"| {' | '.join(cells)} |")
    return rows


def render_markdown_report(evaluation: dict[str, Any], manifest: dict[str, Any]) -> str:
    metrics = evaluation["test_evaluation"]
    confusion = metrics["confusion_matrix"]
    candidates = manifest.get("candidate_evaluation", [])
    importance = manifest.get("feature_importance", [])
    dataset_manifest = manifest.get("dataset", {}).get("manifest", {})
    dataset_source = dataset_manifest.get("source", {})
    source_kind = dataset_source.get("kind", "unknown")
    source_description = (
        "the bundled oversampled development fixture"
        if source_kind == "development_fixture"
        else "a user-supplied public PaySim CSV"
        if source_kind == "public_paysim_csv"
        else "the recorded prepared dataset"
    )
    lines = [
        "# Supervised Model Evaluation",
        "",
        (
            "This report evaluates the frozen selected classifier on the chronological test "
            "partition. Model choice and the operating threshold were fixed using validation "
            "data before test metrics were computed. Accuracy is intentionally not an optimization "
            "target. This is prototype decision support, not evidence for autonomous lending "
        "denial."
        ),
        "",
        "## Data evidence",
        "",
        f"- Prepared source: `{source_kind}` ({source_description})",
        f"- Prepared content SHA-256: `{dataset_manifest.get('content_sha256', 'unknown')}`",
        f"- Rows: `{dataset_manifest.get('row_count', 'unknown')}`",
        "- Lending, device, network, login, and graph fields are deterministic synthetic "
        "enrichment and are not represented as observed PaySim fields.",
        "",
        "## Selection",
        "",
        f"- Model version: `{evaluation['model_version']}`",
        f"- Selected classifier: `{evaluation['model_name']}`",
        f"- Feature schema: `{evaluation['feature_schema_version']}`",
        f"- Frozen threshold: `{_display(metrics['threshold'], digits=6)}`",
        f"- Threshold status: `{manifest.get('selection_status', 'unknown')}`",
        f"- Selection policy: {manifest.get('selection_policy', 'not recorded')}",
        "",
        "The threshold maximizes fraud recall subject to the configured false-positive/review "
        "capacity constraint. Among equally sensitive feasible boundaries, the lowest threshold "
        "is selected to avoid placing the boundary directly on the weakest validation fraud "
        "score. If a constraint cannot be measured on a single-class validation fixture, the "
        "manifest says so rather than reporting a fabricated score.",
        "",
        "### Validation model comparison",
        "",
        "| Model | Threshold | Feasible | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR "
        "| Review rate |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        item = candidate["validation_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate["model_name"],
                    _display(candidate["selected_threshold"], digits=6),
                    _display(candidate["constraints_satisfied"]),
                    _display(item["precision"]),
                    _display(item["recall"]),
                    _display(item["f1"]),
                    _display(item["roc_auc"]),
                    _display(item["pr_auc"]),
                    _display(item["false_positive_rate"]),
                    _display(item["review_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Untouched test performance", "", *_metric_table(metrics), ""])
    undefined = metrics.get("undefined_metrics", [])
    if undefined:
        lines.extend(
            [
                "The following metrics are undefined for this partition and are shown as `N/A`: "
                + ", ".join(undefined)
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            "### Confusion matrix",
            "",
            "| Actual / predicted | Non-fraud | Fraud/review |",
            "|---|---:|---:|",
            f"| Non-fraud | {confusion['tn']} | {confusion['fp']} |",
            f"| Fraud | {confusion['fn']} | {confusion['tp']} |",
            "",
            "### Precision-recall curve data",
            "",
            *_curve_table(
                metrics.get("precision_recall_curve"), ("thresholds", "precision", "recall")
            ),
            "",
            "### ROC curve data",
            "",
            *_curve_table(
                metrics.get("roc_curve"),
                ("thresholds", "false_positive_rate", "true_positive_rate"),
            ),
            "",
            "## Validation threshold analysis",
            "",
        ]
    )
    for candidate in candidates:
        lines.extend(
            [
                f"### {candidate['model_name']}",
                "",
                "| Threshold | Feasible | Precision | Recall | F1 | FPR | Review rate | TP | FP "
                "| TN | FN |",
                "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in candidate.get("threshold_analysis", []):
            matrix = row["confusion_matrix"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        _display(row["threshold"], digits=6),
                        _display(row["feasible"]),
                        _display(row["precision"]),
                        _display(row["recall"]),
                        _display(row["f1"]),
                        _display(row["false_positive_rate"]),
                        _display(row["review_rate"]),
                        str(matrix["tp"]),
                        str(matrix["fp"]),
                        str(matrix["tn"]),
                        str(matrix["fn"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Feature importance",
            "",
            "Absolute standardized coefficients are shown for logistic regression; XGBoost uses "
            "its fitted tree importance. Importance is associative, not causal.",
            "",
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
        ]
    )
    for rank, item in enumerate(importance, start=1):
        lines.append(f"| {rank} | `{item['feature']}` | {_display(item['importance'], digits=6)} |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            f"Results use {source_description}; point estimates are not deployment evidence. "
            "PaySim is transaction simulation data, while lending/device attributes are "
            "deterministic synthetic enrichment. "
            "A production study needs representative labels, temporal drift monitoring, calibrated "
            "review capacity, subgroup error analysis, and human review of false positives.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path("artifacts/supervised-v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("reports/model_evaluation.md"))
    parser.add_argument("--json", action="store_true", help="Print complete evaluation JSON")
    args = parser.parse_args(argv)
    evaluation = evaluate_bundle(args.bundle_dir, args.data_dir)
    model = load_classifier_bundle(args.bundle_dir)
    report = render_markdown_report(evaluation, model.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    if args.json:
        print(json.dumps(evaluation, indent=2, sort_keys=True, allow_nan=False))
    else:
        metrics = evaluation["test_evaluation"]
        print(
            json.dumps(
                {
                    "report": str(args.output),
                    "model_version": evaluation["model_version"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "false_positive_rate": metrics["false_positive_rate"],
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
