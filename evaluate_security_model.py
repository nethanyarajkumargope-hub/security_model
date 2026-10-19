"""Evaluate the frozen security model on a labelled flow CSV without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from security_risk import attack_risk, threshold_risk
from train_security_model import metrics
from verify_security_release import load_verified_model


def evaluate_frame(frame: pd.DataFrame, artifact: dict) -> dict:
    if frame.empty:
        raise ValueError("Evaluation data is empty")
    label_column = artifact.get("label_column", "label")
    attack_column = artifact.get("attack_column", "attack_cat")
    if label_column not in frame:
        raise ValueError(f"Evaluation data is missing {label_column}")
    labels = pd.to_numeric(frame[label_column], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Evaluation labels must contain only 0 and 1")
    if set(labels.unique()) != {0, 1}:
        raise ValueError("Evaluation data must include both label 0 and label 1")

    labels = labels.to_numpy(dtype=int)
    risk = attack_risk(frame, artifact)
    threshold = float(artifact["threshold"])
    report = {
        "rows": int(len(frame)),
        "selected_model": artifact.get("selected_model"),
        "threshold": threshold,
        "metrics": metrics(labels, risk, threshold),
        "attack_categories": {},
    }
    if attack_column in frame:
        normalized = frame.reset_index(drop=True)
        decision = threshold_risk(risk, threshold)
        for category, indices in normalized.groupby(attack_column, dropna=False).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            report["attack_categories"][str(category)] = {
                "support": int(len(positions)),
                "mean_risk": float(risk[positions].mean()),
                "predicted_attack_rate": float(decision[positions].mean()),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_json", nargs="?", default="artifacts/security/evaluation.json")
    parser.add_argument("--model", default="artifacts/security/security_model.joblib")
    parser.add_argument("--manifest")
    args = parser.parse_args()

    artifact = load_verified_model(args.model, args.manifest)
    report = evaluate_frame(pd.read_csv(args.input_csv, low_memory=False), artifact)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
