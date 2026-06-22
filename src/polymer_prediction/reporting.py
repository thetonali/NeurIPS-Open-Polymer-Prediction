from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from polymer_prediction.config import ProjectConfig
from polymer_prediction.data import data_profile, label_stats
from polymer_prediction.features import RDKIT_AVAILABLE


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_oof_predictions(
    train: pd.DataFrame, oof: pd.DataFrame, config: ProjectConfig, output_dir: Path
) -> Path:
    output = pd.DataFrame(index=train.index)
    if config.id_column in train.columns:
        output[config.id_column] = train[config.id_column].to_numpy()
    if config.smiles_column in train.columns:
        output[config.smiles_column] = train[config.smiles_column].to_numpy()
    for target in config.target_columns:
        output[f"{target}_true"] = train[target].to_numpy()
        output[f"{target}_oof"] = oof[target].to_numpy()

    parquet_path = output_dir / "oof_predictions.parquet"
    try:
        output.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = output_dir / "oof_predictions.csv"
        output.to_csv(csv_path, index=False)
        return csv_path


def write_baseline_artifacts(
    train: pd.DataFrame,
    oof: pd.DataFrame,
    cv_result: dict[str, Any],
    config: ProjectConfig,
    used_supplements: bool,
) -> dict[str, Path]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = label_stats(train, config.target_columns)
    label_stats_path = output_dir / "label_stats.csv"
    stats.to_csv(label_stats_path, index=False)

    oof_path = save_oof_predictions(train, oof, config, output_dir)

    payload = {
        "config": config.to_dict(),
        "seed": config.seed,
        "n_splits": config.n_splits,
        "data": data_profile(train, config.target_columns),
        "used_supplements": used_supplements,
        "fold_metrics": cv_result["fold_metrics"],
        "target_summary": cv_result["target_summary"],
        "scores": cv_result["scores"],
        "overall_metric": cv_result["overall_metric"],
    }
    baseline_path = output_dir / "baseline_cv.json"
    baseline_path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )

    summary_path = output_dir / "experiment_summary.md"
    summary_path.write_text(
        build_experiment_summary(train, stats, cv_result, config, used_supplements),
        encoding="utf-8",
    )

    return {
        "baseline_cv": baseline_path,
        "label_stats": label_stats_path,
        "oof_predictions": oof_path,
        "experiment_summary": summary_path,
    }


def build_experiment_summary(
    train: pd.DataFrame,
    stats: pd.DataFrame,
    cv_result: dict[str, Any],
    config: ProjectConfig,
    used_supplements: bool,
) -> str:
    feature_sets = config.feature_sets or {
        "smiles_tfidf": True,
        "basic_smiles_statistics": True,
        "rdkit_descriptors": RDKIT_AVAILABLE,
    }
    model_type = config.model.get("type", "ridge")

    lines = [
        "# Baseline Experiment Summary",
        "",
        "## Data Scale",
        "",
        f"- Training rows used: {len(train)}",
        f"- Columns: {', '.join(map(str, train.columns))}",
        f"- Supplements used: {'yes' if used_supplements else 'no'}",
        f"- Sample size: {config.sample_size if config.sample_size is not None else 'full data'}",
        "",
        "## Label Missingness",
        "",
        "| Target | Non-missing | Missing | Missing rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in stats.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.non_missing_count} | {row.missing_count} | "
            f"{row.missing_rate:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Feature Setting",
            "",
        ]
    )
    for name, enabled in feature_sets.items():
        lines.append(f"- {name}: {enabled}")
    lines.append(f"- RDKit available in this run: {RDKIT_AVAILABLE}")

    lines.extend(
        [
            "",
            "## Model Setting",
            "",
            f"- Model type: {model_type}",
            "- Training strategy: one independent model per target",
            f"- Validation: KFold CV, requested n_splits={config.n_splits}",
            f"- Seed: {config.seed}",
            "",
            "## CV Results",
            "",
            "| Target | Fold mean MAE | Fold std MAE | OOF MAE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for target in config.target_columns:
        summary = cv_result["target_summary"].get(target, {})
        lines.append(
            f"| {target} | {summary.get('mean', float('nan')):.6f} | "
            f"{summary.get('std', float('nan')):.6f} | "
            f"{summary.get('oof_mae', float('nan')):.6f} |"
        )
    lines.append(f"\nOverall weighted MAE estimate: {cv_result['overall_metric']:.6f}")

    lines.extend(
        [
            "",
            "## Current Limitations",
            "",
            "- This is a reproducible academic baseline, not a leaderboard-optimized system.",
            "- Validation is plain K-fold and does not yet account for molecular similarity groups.",
            "- Feature ablations have not yet been run.",
            "- Morgan fingerprints and model comparisons are planned but intentionally not included in Phase 1.",
        ]
    )

    return "\n".join(lines) + "\n"
