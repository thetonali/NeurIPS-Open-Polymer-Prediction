from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from polymer_prediction.config import ProjectConfig
from polymer_prediction.features import FeatureBuilder
from polymer_prediction.metrics import estimate_competition_weights, property_mae, weighted_mae


def build_regressor(model_config: dict[str, Any], random_state: int):
    model_type = model_config.get("type", "hist_gradient_boosting")
    if model_type == "ridge":
        return Ridge(alpha=float(model_config.get("alpha", 5.0)), random_state=random_state)
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(model_config.get("n_estimators", 500)),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 2)),
            n_jobs=-1,
            random_state=random_state,
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=int(model_config.get("max_iter", 700)),
            learning_rate=float(model_config.get("learning_rate", 0.045)),
            l2_regularization=float(model_config.get("l2_regularization", 0.02)),
            max_leaf_nodes=int(model_config.get("max_leaf_nodes", 31)),
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def build_pipeline(config: ProjectConfig):
    features = FeatureBuilder(config.tfidf).build()
    regressor = build_regressor(config.model, config.random_state)
    return Pipeline([("features", features), ("regressor", regressor)])


def _effective_n_splits(non_missing_count: int, requested_splits: int, target: str) -> int:
    if non_missing_count < 2:
        raise ValueError(
            f"Target {target} has only {non_missing_count} labeled rows; "
            "at least 2 are required for K-fold CV."
        )
    return min(requested_splits, non_missing_count)


def cross_validate_detailed(train: pd.DataFrame, config: ProjectConfig) -> dict[str, Any]:
    oof = pd.DataFrame(index=train.index, columns=config.target_columns, dtype=float)
    weights = estimate_competition_weights(train, config.target_columns)
    fold_metrics: dict[str, list[dict[str, float | int]]] = {}

    for target in config.target_columns:
        target_mask = train[target].notna()
        target_train = train.loc[target_mask].reset_index()
        n_splits = _effective_n_splits(len(target_train), config.n_splits, target)

        if n_splits < config.n_splits:
            print(f"{target}: reducing n_splits from {config.n_splits} to {n_splits}")

        fold_metrics[target] = []
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
        for fold, (fit_idx, valid_idx) in enumerate(kfold.split(target_train), start=1):
            fit_frame = target_train.iloc[fit_idx]
            valid_frame = target_train.iloc[valid_idx]
            model = build_pipeline(config)
            model.fit(fit_frame[[config.smiles_column]], fit_frame[target])
            pred = model.predict(valid_frame[[config.smiles_column]])
            oof.loc[valid_frame["index"].to_numpy(), target] = pred
            mae = float(np.mean(np.abs(pred - valid_frame[target].to_numpy())))
            fold_metrics[target].append(
                {
                    "fold": fold,
                    "n_train": int(len(fit_frame)),
                    "n_valid": int(len(valid_frame)),
                    "mae": mae,
                }
            )
            print(f"{target} fold {fold}: MAE={mae:.6f}")

    mae_by_target = property_mae(train[config.target_columns], oof, config.target_columns)
    overall_metric = weighted_mae(
        train[config.target_columns], oof, config.target_columns, weights
    )
    mae_by_target["weighted_mae_estimate"] = overall_metric

    target_summary = {}
    for target, rows in fold_metrics.items():
        values = np.array([row["mae"] for row in rows], dtype=float)
        target_summary[target] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "oof_mae": mae_by_target.get(target),
        }

    return {
        "oof": oof,
        "scores": mae_by_target,
        "fold_metrics": fold_metrics,
        "target_summary": target_summary,
        "overall_metric": overall_metric,
    }


def cross_validate(train: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    result = cross_validate_detailed(train, config)
    oof = result["oof"]
    mae_by_target = result["scores"]
    return oof, mae_by_target


def train_full_models(train: pd.DataFrame, config: ProjectConfig) -> dict[str, Any]:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    for target in config.target_columns:
        mask = train[target].notna()
        if not mask.any():
            raise ValueError(f"No labels found for target: {target}")
        model = build_pipeline(config)
        model.fit(train.loc[mask, [config.smiles_column]], train.loc[mask, target])
        models[target] = model
        joblib.dump(model, config.model_dir / f"{target}.joblib")
    return models


def load_models(model_dir: Path, targets: list[str]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for target in targets:
        path = model_dir / f"{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing trained model: {path}")
        models[target] = joblib.load(path)
    return models


def predict(models: dict[str, Any], test: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    pred = pd.DataFrame({config.id_column: test[config.id_column].to_numpy()})
    for target in config.target_columns:
        pred[target] = models[target].predict(test[[config.smiles_column]])
    return pred
