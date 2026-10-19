"""Reusable security-risk contract compatible with the reference simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_input(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Input CSV has duplicate columns: {duplicates}")
    required = artifact["feature_columns"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing model features: {sorted(missing)}")
    prepared = frame.loc[:, required].copy()
    categorical = set(artifact.get("categorical_columns", []))
    for column in set(required) - categorical:
        try:
            prepared[column] = pd.to_numeric(prepared[column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Input feature {column!r} must be numeric") from error
    return prepared.replace([np.inf, -np.inf], np.nan)


def attack_risk(frame: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Return calibrated P(label=1), matching the reference simulator's risk contract."""
    estimator = artifact["estimator"]
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("Model does not expose the attack class (label 1)")
    probability = estimator.predict_proba(prepare_input(frame, artifact))[:, classes.index(1)]
    return np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)


def threshold_risk(risk, threshold: float) -> np.ndarray:
    return (np.clip(np.asarray(risk, dtype=float), 0.0, 1.0) >= threshold).astype(int)
