"""Measure local batch inference latency and peak Python allocations."""

from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from security_risk import prepare_input
from verify_security_release import load_verified_model


def measure_batch(estimator, frame: pd.DataFrame, repeats: int) -> dict:
    estimator.predict_proba(frame)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        estimator.predict_proba(frame)
        durations.append(time.perf_counter() - start)

    gc.collect()
    tracemalloc.start()
    estimator.predict_proba(frame)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    median_seconds = float(np.median(durations))
    return {
        "rows": int(len(frame)),
        "repeats": repeats,
        "median_seconds": median_seconds,
        "milliseconds_per_row": float(1000 * median_seconds / len(frame)),
        "rows_per_second": float(len(frame) / median_seconds),
        "peak_python_allocation_bytes": int(peak_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/security/security_model.joblib")
    parser.add_argument("--manifest")
    parser.add_argument("--input", default="data/unsw_nb15/UNSW_NB15_testing-set.csv")
    parser.add_argument("--output", default="artifacts/security/inference_benchmark.json")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    artifact = load_verified_model(args.model, args.manifest)
    frame = prepare_input(pd.read_csv(args.input, low_memory=False), artifact)
    batches = [measure_batch(artifact["estimator"], frame.iloc[:size], args.repeats)
               for size in (1, 100, 1000)]
    report = {
        "model": artifact["selected_model"],
        "threshold": artifact["threshold"],
        "model_file_bytes": Path(args.model).stat().st_size,
        "batches": batches,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
