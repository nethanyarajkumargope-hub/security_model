"""Apply the trained UNSW-NB15 binary security model to a CSV file."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--model", default="artifacts/security/security_model.joblib")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    frame = pd.read_csv(args.input_csv, low_memory=False)
    missing = set(artifact["feature_columns"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing model features: {sorted(missing)}")
    probability = artifact["estimator"].predict_proba(frame)[:, 1]
    output = pd.DataFrame({"attack_probability": probability,
                           "predicted_label": (probability >= artifact["threshold"]).astype(int)})
    if "id" in frame:
        output.insert(0, "id", frame["id"])
    output.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
