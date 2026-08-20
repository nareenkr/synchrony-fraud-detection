"""Train the deterministic anomaly component on replayed canonical features."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from backend.app.fraud.anomaly import DEFAULT_ANOMALY_CONFIG, AnomalyConfig, AnomalyScorer
from training.prepare_data import DatasetSplits
from training.replay import ModelMatrixSplits, replay_splits


def load_prepared_splits(data_dir: str | Path) -> DatasetSplits:
    directory = Path(data_dir)
    partitions = {}
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"prepared partition not found: {path}")
        partitions[name] = pd.read_csv(path)
    return DatasetSplits(**partitions)


def train_anomaly(
    splits: DatasetSplits,
    *,
    config: AnomalyConfig = DEFAULT_ANOMALY_CONFIG,
    normal_only: bool = True,
) -> tuple[AnomalyScorer, ModelMatrixSplits]:
    """Replay partitions and fit solely on the chronological training matrix."""

    matrices = replay_splits(splits)
    training_features = matrices.train.features
    if normal_only:
        training_features = training_features.loc[matrices.train.labels == 0]
    if len(training_features) < 2:
        raise ValueError("not enough training observations for anomaly detection")
    return AnomalyScorer(config).fit(training_features), matrices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/anomaly.joblib"))
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--include-fraud",
        action="store_true",
        help="Fit on every training row instead of learning the normal-only profile.",
    )
    args = parser.parse_args(argv)
    config = AnomalyConfig(
        n_estimators=args.estimators,
        max_samples=args.max_samples,
        random_state=args.seed,
    )
    scorer, matrices = train_anomaly(
        load_prepared_splits(args.data_dir),
        config=config,
        normal_only=not args.include_fraud,
    )
    digest = scorer.save(args.output)
    summary = {
        "artifact": str(args.output),
        "sha256": digest,
        "fit_partition": "train",
        "normal_only": not args.include_fraud,
        "train_rows": len(matrices.train.features),
        "validation_rows": len(matrices.validation.features),
        "test_rows": len(matrices.test.features),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
