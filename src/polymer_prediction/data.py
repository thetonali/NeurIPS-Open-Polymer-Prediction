from __future__ import annotations

from pathlib import Path

import pandas as pd

from polymer_prediction.config import ProjectConfig


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_competition_data(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = read_csv(config.train_path)
    test = read_csv(config.test_path)
    sample_submission = read_csv(config.sample_submission_path)
    return train, test, sample_submission


def load_supplement_data(config: ProjectConfig) -> list[pd.DataFrame]:
    if not config.supplement_path.exists():
        return []

    frames: list[pd.DataFrame] = []
    for path in sorted(config.supplement_path.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    return frames


def normalize_target_columns(frame: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for target in target_columns:
        if target not in out.columns:
            out[target] = pd.NA
    return out


def merge_train_and_supplements(train: pd.DataFrame, supplements: list[pd.DataFrame], config: ProjectConfig) -> pd.DataFrame:
    """Append supplement rows that contain SMILES and at least one known target.

    Supplement files in this competition are heterogeneous. This conservative
    merge keeps only columns needed by the training pipeline and ignores files
    that provide SMILES without labels.
    """

    frames = [normalize_target_columns(train, config.target_columns)]
    required = {config.smiles_column}

    for supplement in supplements:
        if not required.issubset(supplement.columns):
            continue
        supplement = normalize_target_columns(supplement, config.target_columns)
        has_label = supplement[config.target_columns].notna().any(axis=1)
        if not has_label.any():
            continue
        keep_columns = [config.smiles_column, *config.target_columns]
        if config.id_column in supplement.columns:
            keep_columns.insert(0, config.id_column)
        frames.append(supplement.loc[has_label, keep_columns])

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=[config.smiles_column, *config.target_columns])
    return merged
