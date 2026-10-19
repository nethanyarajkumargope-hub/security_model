"""Check security-model selection and threshold stability across development seeds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from train_security_model import candidates, features, grouped_split, metrics, select_threshold


def run_seed(frame: pd.DataFrame, seed: int) -> dict:
    fit_index, validation_index, calibration_index, threshold_index = grouped_split(frame, seed)
    x, y = features(frame), frame["label"].to_numpy(dtype=int)
    validation, fitted = {}, {}
    for name, model in candidates(x, seed).items():
        model.fit(x.iloc[fit_index], y[fit_index])
        validation[name] = metrics(
            y[validation_index], model.predict_proba(x.iloc[validation_index])[:, 1]
        )
        fitted[name] = model

    selected = max(validation, key=lambda name: (
        validation[name]["balanced_accuracy"], validation[name]["f1"]
    ))
    final_pipeline = clone(fitted[selected])
    development_index = np.concatenate([fit_index, validation_index])
    final_pipeline.fit(x.iloc[development_index], y[development_index])
    calibrated = CalibratedClassifierCV(FrozenEstimator(final_pipeline), method="sigmoid")
    calibrated.fit(x.iloc[calibration_index], y[calibration_index])
    probability = calibrated.predict_proba(x.iloc[threshold_index])[:, 1]
    threshold = select_threshold(y[threshold_index], probability)
    return {
        "seed": seed,
        "selected_model": selected,
        "threshold": threshold,
        "selected_validation": validation[selected],
        "validation_candidates": validation,
        "threshold_tuning": metrics(y[threshold_index], probability, threshold),
    }


def summarize_runs(runs: list[dict]) -> dict:
    def summary(section: str, metric: str) -> dict:
        values = np.asarray([run[section][metric] for run in runs], dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std()),
                "minimum": float(values.min()), "maximum": float(values.max())}

    thresholds = np.asarray([run["threshold"] for run in runs], dtype=float)
    return {
        "selected_models": dict(Counter(run["selected_model"] for run in runs)),
        "threshold": {"mean": float(thresholds.mean()), "std": float(thresholds.std()),
                      "minimum": float(thresholds.min()), "maximum": float(thresholds.max())},
        "validation_balanced_accuracy": summary("selected_validation", "balanced_accuracy"),
        "threshold_tuning_recall": summary("threshold_tuning", "recall"),
        "threshold_tuning_false_positive_rate": summary("threshold_tuning", "false_positive_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/unsw_nb15/UNSW_NB15_training-set.csv")
    parser.add_argument("--output", default="artifacts/security/multiseed.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    args = parser.parse_args()

    frame = pd.read_csv(args.train, low_memory=False)
    runs = [run_seed(frame, seed) for seed in args.seeds]
    report = {"seeds": args.seeds, "runs": runs, "summary": summarize_runs(runs)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
