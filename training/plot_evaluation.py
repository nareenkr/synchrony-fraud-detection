"""Render retained model-evaluation plots from the frozen bundle and test split."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backend.app.explainability import ShapContributionAdapter
from backend.app.fraud.supervised import load_classifier_bundle
from training.replay import replay_splits
from training.train_classifier import load_prepared_splits


def _save(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def generate_plots(
    bundle_dir: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    model = load_classifier_bundle(bundle_dir)
    manifest = model.manifest
    metrics = manifest["test_evaluation"]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    confusion = metrics["confusion_matrix"]
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], ["Non-fraud", "Flagged"])
    axis.set_yticks([0, 1], ["Non-fraud", "Fraud"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Frozen-threshold confusion matrix")
    figure.colorbar(image, ax=axis)
    path = destination / "confusion_matrix.png"
    _save(figure, path)
    paths.append(path)

    roc = metrics.get("roc_curve")
    figure, axis = plt.subplots(figsize=(5, 4))
    if roc:
        axis.plot(roc["false_positive_rate"], roc["true_positive_rate"], label="Classifier")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Random")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axis.legend()
    path = destination / "roc_curve.png"
    _save(figure, path)
    paths.append(path)

    precision_recall = metrics.get("precision_recall_curve")
    figure, axis = plt.subplots(figsize=(5, 4))
    if precision_recall:
        axis.plot(
            precision_recall["recall"],
            precision_recall["precision"],
            label="Classifier",
        )
    prevalence = metrics["positive_count"] / metrics["row_count"]
    axis.axhline(prevalence, linestyle="--", color="#64748b", label="Prevalence")
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axis.legend()
    path = destination / "precision_recall_curve.png"
    _save(figure, path)
    paths.append(path)

    selected = next(
        candidate
        for candidate in manifest["candidate_evaluation"]
        if candidate["model_name"] == manifest["model_name"]
    )
    table = selected["threshold_analysis"]
    thresholds = [row["threshold"] for row in table]
    recalls = [np.nan if row["recall"] is None else row["recall"] for row in table]
    false_positive_rates = [
        np.nan if row["false_positive_rate"] is None else row["false_positive_rate"]
        for row in table
    ]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(thresholds, recalls, label="Recall")
    axis.plot(thresholds, false_positive_rates, label="False-positive rate")
    axis.axvline(model.threshold, color="#ef4444", linestyle="--", label="Selected")
    axis.set(xlabel="Probability threshold", ylabel="Rate", title="Validation threshold analysis")
    axis.legend()
    path = destination / "threshold_analysis.png"
    _save(figure, path)
    paths.append(path)

    importance = list(manifest.get("feature_importance", ()))[:15]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.barh(
        [item["feature"] for item in reversed(importance)],
        [item["importance"] for item in reversed(importance)],
        color="#0f766e",
    )
    axis.set(xlabel="Importance", title="Selected-model feature importance")
    path = destination / "feature_importance.png"
    _save(figure, path)
    paths.append(path)

    matrices = replay_splits(load_prepared_splits(data_dir))
    probabilities = model.predict_proba(matrices.test.features)
    representative = matrices.test.features.iloc[[int(np.argmax(probabilities))]]
    contributions = ShapContributionAdapter().explain(model.estimator, representative)
    ranked = sorted(contributions, key=lambda item: abs(item.contribution), reverse=True)[:10]
    figure, axis = plt.subplots(figsize=(7, 5))
    names = [item.feature.rsplit("__", 1)[-1] for item in reversed(ranked)]
    values = [item.contribution for item in reversed(ranked)]
    colors = ["#dc2626" if value > 0 else "#2563eb" for value in values]
    axis.barh(names, values, color=colors)
    axis.axvline(0, color="#334155", linewidth=0.8)
    axis.set(xlabel="SHAP contribution (internal evaluation only)", title="Local SHAP explanation")
    path = destination / "shap_explanation.png"
    _save(figure, path)
    paths.append(path)
    return tuple(paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path("artifacts/supervised-v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/plots"))
    args = parser.parse_args(argv)
    for path in generate_plots(args.bundle_dir, args.data_dir, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
