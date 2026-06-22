from __future__ import annotations

from pathlib import Path

import pandas as pd

from polymer_prediction.config import ProjectConfig


def read_table(path: str | Path, sample_size: int | None = None) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required data file does not exist: {path}. "
            "Check train_path/test_path/sample_submission_file in config/default.json."
        )

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if sample_size is None:
            return pd.read_parquet(path)

        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            batch = next(parquet.iter_batches(batch_size=int(sample_size)))
            return batch.to_pandas()
        except Exception:
            return pd.read_parquet(path).head(int(sample_size))

    if suffix == ".csv":
        return pd.read_csv(path, nrows=sample_size)

    raise ValueError(f"Unsupported file format for {path}. Expected .csv or .parquet.")


def label_stats(train: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    total_rows = len(train)
    for target in target_columns:
        if target not in train.columns:
            missing_count = total_rows
            non_missing_count = 0
        else:
            missing_count = int(train[target].isna().sum())
            non_missing_count = int(train[target].notna().sum())
        rows.append(
            {
                "target": target,
                "rows": total_rows,
                "missing_count": missing_count,
                "missing_rate": float(missing_count / total_rows) if total_rows else 0.0,
                "non_missing_count": non_missing_count,
            }
        )
    return pd.DataFrame(rows)


def data_profile(train: pd.DataFrame, target_columns: list[str]) -> dict[str, object]:
    stats = label_stats(train, target_columns)
    return {
        "n_rows": int(len(train)),
        "columns": list(train.columns),
        "target_missing_rate": dict(zip(stats["target"], stats["missing_rate"])),
        "target_non_missing_count": dict(zip(stats["target"], stats["non_missing_count"])),
    }


def print_data_profile(name: str, frame: pd.DataFrame, target_columns: list[str]) -> None:
    print(f"{name} rows: {len(frame)}")
    print(f"{name} columns: {list(frame.columns)}")
    if set(target_columns).intersection(frame.columns):
        stats = label_stats(frame, target_columns)
        print(f"{name} target missing rates:")
        for row in stats.itertuples(index=False):
            print(
                f"  {row.target}: missing_rate={row.missing_rate:.4f}, "
                f"non_missing={row.non_missing_count}"
            )


def load_competition_data(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_size = getattr(config, "sample_size", None)

    train = read_table(config.train_path, sample_size=sample_size)
    test = read_table(config.test_path, sample_size=sample_size)
    sample_submission = read_table(config.sample_submission_path)

    print_data_profile("train", train, config.target_columns)
    print_data_profile("test", test, config.target_columns)

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


def merge_train_and_supplements(
    train: pd.DataFrame, supplements: list[pd.DataFrame], config: ProjectConfig
) -> pd.DataFrame:
    """Append supplement rows that contain SMILES and at least one known target."""

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
