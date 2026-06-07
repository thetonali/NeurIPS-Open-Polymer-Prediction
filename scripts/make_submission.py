from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polymer_prediction.config import load_config
from polymer_prediction.data import load_competition_data, load_supplement_data, merge_train_and_supplements
from polymer_prediction.modeling import load_models, predict, train_full_models
from polymer_prediction.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.random_state)

    train, test, sample_submission = load_competition_data(config)
    if args.train_if_missing:
        supplements = load_supplement_data(config)
        train = merge_train_and_supplements(train, supplements, config)
        train_full_models(train, config)

    models = load_models(config.model_dir, config.target_columns)
    submission = predict(models, test, config)
    submission = sample_submission[[config.id_column]].merge(submission, on=config.id_column, how="left")
    submission = submission[[config.id_column, *config.target_columns]]

    output_path = Path(args.output) if args.output else config.submission_dir / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission: {output_path}")


if __name__ == "__main__":
    main()
