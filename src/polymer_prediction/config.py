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
    sample_submission_file: str
    supplement_dir: str
    model_dir: Path
    submission_dir: Path
    target_columns: list[str]
    id_column: str
    smiles_column: str
    n_splits: int
    random_state: int
    tfidf: dict[str, Any]
    model: dict[str, Any]

    @property
    def train_path(self) -> Path:
        return self.data_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.data_dir / self.test_file

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / self.sample_submission_file

    @property
    def supplement_path(self) -> Path:
        return self.data_dir / self.supplement_dir


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return ProjectConfig(
        data_dir=Path(raw["data_dir"]),
        train_file=raw["train_file"],
        test_file=raw["test_file"],
        sample_submission_file=raw["sample_submission_file"],
        supplement_dir=raw["supplement_dir"],
        model_dir=Path(raw["model_dir"]),
        submission_dir=Path(raw["submission_dir"]),
        target_columns=list(raw["target_columns"]),
        id_column=raw["id_column"],
        smiles_column=raw["smiles_column"],
        n_splits=int(raw["n_splits"]),
        random_state=int(raw["random_state"]),
        tfidf=dict(raw["tfidf"]),
        model=dict(raw["model"]),
    )
