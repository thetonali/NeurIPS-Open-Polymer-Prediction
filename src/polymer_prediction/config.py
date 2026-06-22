from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: Path
    train_file: str
    test_file: str
    train_path: Path
    test_path: Path
    sample_submission_file: str
    supplement_dir: str
    model_dir: Path
    submission_dir: Path
    output_dir: Path
    target_columns: list[str]
    id_column: str
    smiles_column: str
    seed: int
    n_splits: int
    random_state: int
    sample_size: int | None
    feature_sets: dict[str, Any]
    tfidf: dict[str, Any]
    model: dict[str, Any]

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / self.sample_submission_file

    @property
    def supplement_path(self) -> Path:
        return self.data_dir / self.supplement_dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "train_file": self.train_file,
            "test_file": self.test_file,
            "train_path": str(self.train_path),
            "test_path": str(self.test_path),
            "sample_submission_file": self.sample_submission_file,
            "supplement_dir": self.supplement_dir,
            "model_dir": str(self.model_dir),
            "submission_dir": str(self.submission_dir),
            "output_dir": str(self.output_dir),
            "target_columns": self.target_columns,
            "id_column": self.id_column,
            "smiles_column": self.smiles_column,
            "seed": self.seed,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "sample_size": self.sample_size,
            "feature_sets": self.feature_sets,
            "tfidf": self.tfidf,
            "model": self.model,
        }

    def with_sample_size(self, sample_size: int | None) -> "ProjectConfig":
        return ProjectConfig(
            data_dir=self.data_dir,
            train_file=self.train_file,
            test_file=self.test_file,
            train_path=self.train_path,
            test_path=self.test_path,
            sample_submission_file=self.sample_submission_file,
            supplement_dir=self.supplement_dir,
            model_dir=self.model_dir,
            submission_dir=self.submission_dir,
            output_dir=self.output_dir,
            target_columns=self.target_columns,
            id_column=self.id_column,
            smiles_column=self.smiles_column,
            seed=self.seed,
            n_splits=self.n_splits,
            random_state=self.random_state,
            sample_size=sample_size,
            feature_sets=self.feature_sets,
            tfidf=self.tfidf,
            model=self.model,
        )


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    data_dir = Path(raw.get("data_dir", "data/raw"))
    train_file = raw.get("train_file", Path(raw.get("train_path", "train.parquet")).name)
    test_file = raw.get("test_file", Path(raw.get("test_path", "test.parquet")).name)
    train_path = Path(raw.get("train_path", data_dir / train_file))
    test_path = Path(raw.get("test_path", data_dir / test_file))
    seed = int(raw.get("seed", raw.get("random_state", 42)))

    return ProjectConfig(
        data_dir=data_dir,
        train_file=train_file,
        test_file=test_file,
        train_path=train_path,
        test_path=test_path,
        sample_submission_file=raw.get("sample_submission_file", "sample_submission.csv"),
        supplement_dir=raw.get("supplement_dir", "train_supplement"),
        model_dir=Path(raw.get("model_dir", "models")),
        submission_dir=Path(raw.get("submission_dir", "data/submissions")),
        output_dir=Path(raw.get("output_dir", "reports")),
        target_columns=list(raw["target_columns"]),
        id_column=raw.get("id_column", "id"),
        smiles_column=raw.get("smiles_column", "SMILES"),
        seed=seed,
        n_splits=int(raw.get("n_splits", 5)),
        random_state=int(raw.get("random_state", seed)),
        sample_size=raw.get("sample_size"),
        feature_sets=dict(raw.get("feature_sets", {})),
        tfidf=dict(raw.get("tfidf", {})),
        model=dict(raw.get("model", {})),
    )
