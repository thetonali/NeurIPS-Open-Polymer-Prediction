from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polymer_prediction.config import load_config
from polymer_prediction.data import load_competition_data, load_supplement_data, merge_train_and_supplements
from polymer_prediction.modeling import cross_validate_detailed, train_full_models
from polymer_prediction.reporting import write_baseline_artifacts
from polymer_prediction.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--no-supplement", action="store_true")
    parser.add_argument("--skip-train-full", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.sample_size is not None:
        config = config.with_sample_size(args.sample_size)
    seed_everything(config.seed)

    train, _, _ = load_competition_data(config)
    used_supplements = False
    if not args.no_supplement:
        supplements = load_supplement_data(config)
        train = merge_train_and_supplements(train, supplements, config)
        used_supplements = bool(supplements)

    print(f"Training rows: {len(train)}")
    print("Label counts:")
    print(train[config.target_columns].notna().sum())

    cv_result = cross_validate_detailed(train, config)
    oof = cv_result["oof"]
    scores = cv_result["scores"]
    print("\nCV scores:")
    for name, score in scores.items():
        print(f"{name}: {score:.6f}")

    artifacts = write_baseline_artifacts(train, oof, cv_result, config, used_supplements)
    print("\nSaved baseline artifacts:")
    for name, path in artifacts.items():
        print(f"{name}: {path}")

    if not args.skip_train_full:
        train_full_models(train, config)
        print(f"\nSaved models to: {config.model_dir}")


if __name__ == "__main__":
    main()
