"""Apply the trained UNSW-NB15 binary security model to a CSV file."""

from __future__ import annotations

import argparse

import pandas as pd

from security_risk import attack_risk, threshold_risk
from verify_security_release import load_verified_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--model", default="artifacts/security/security_model.joblib")
    parser.add_argument("--manifest")
    args = parser.parse_args()

    artifact = load_verified_model(args.model, args.manifest)
    frame = pd.read_csv(args.input_csv, low_memory=False)
    probability = attack_risk(frame, artifact)
    output = pd.DataFrame({"attack_probability": probability,
                           "predicted_label": threshold_risk(probability, artifact["threshold"])})
    if "id" in frame:
        output.insert(0, "id", frame["id"])
    output.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
