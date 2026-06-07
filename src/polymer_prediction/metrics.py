from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def property_mae(y_true: pd.DataFrame, y_pred: pd.DataFrame, targets: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for target in targets:
        mask = y_true[target].notna()
        if mask.any():
            scores[target] = float(mean_absolute_error(y_true.loc[mask, target], y_pred.loc[mask, target]))
    return scores


def estimate_competition_weights(train: pd.DataFrame, targets: list[str]) -> dict[str, float]:
    counts = train[targets].notna().sum(axis=0).astype(float)
    ranges = (train[targets].max(axis=0) - train[targets].min(axis=0)).astype(float)
    ranges = ranges.replace(0, np.nan)

    rarity = np.sqrt(1.0 / counts.replace(0, np.nan))
    normalized_rarity = len(targets) * rarity / rarity.sum()
    weights = (1.0 / ranges) * normalized_rarity
    return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_dict()


def weighted_mae(y_true: pd.DataFrame, y_pred: pd.DataFrame, targets: list[str], weights: dict[str, float]) -> float:
    total = 0.0
    count = 0
    for target in targets:
        mask = y_true[target].notna()
        if not mask.any():
            continue
        errors = np.abs(y_pred.loc[mask, target].to_numpy() - y_true.loc[mask, target].to_numpy())
        total += float(np.sum(weights[target] * errors))
        count += int(mask.sum())
    return total / max(count, 1)
