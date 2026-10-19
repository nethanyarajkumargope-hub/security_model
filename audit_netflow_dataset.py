"""Stream a large labelled NetFlow CSV and report training-data quality."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def audit_chunks(chunks) -> dict:
    rows = duplicates = 0
    columns = None
    label_counts, attack_counts = Counter(), Counter()
    missing, infinite, dtypes, seen_hashes = Counter(), Counter(), {}, set()
    identity_counts = {column: {} for column in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")}
    identity_counts["IP_PAIR"] = {}
    label_attack_mismatches = 0
    for chunk in chunks:
        if columns is None:
            columns = list(chunk.columns)
        elif list(chunk.columns) != columns:
            raise ValueError("CSV schema changed between chunks")
        rows += len(chunk)
        for value, count in chunk["Label"].value_counts(dropna=False).items():
            label_counts[str(value)] += int(count)
        for value, count in chunk["Attack"].value_counts(dropna=False).items():
            attack_counts[str(value)] += int(count)
        label_attack_mismatches += int(
            ((chunk["Label"] == 0) != (chunk["Attack"] == "Benign")).sum()
        )
        for column in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
            for (value, label), count in chunk.groupby([column, "Label"]).size().items():
                totals = identity_counts[column].setdefault(str(value), [0, 0])
                totals[int(label)] += int(count)
        for (source, destination, label), count in chunk.groupby(
            ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label"]
        ).size().items():
            totals = identity_counts["IP_PAIR"].setdefault(f"{source}>{destination}", [0, 0])
            totals[int(label)] += int(count)
        missing.update({column: int(count) for column, count in chunk.isna().sum().items() if count})
        numeric = chunk.select_dtypes(include=[np.number])
        infinite.update({column: int(count) for column, count in np.isinf(numeric).sum().items() if count})
        dtypes.update({column: str(dtype) for column, dtype in chunk.dtypes.items()})
        for value in pd.util.hash_pandas_object(chunk, index=False).to_numpy():
            key = int(value)
            duplicates += key in seen_hashes
            seen_hashes.add(key)
    if columns is None:
        raise ValueError("Dataset is empty")
    identity_summary = {
        name: {
            "unique_values": len(counts),
            "values_seen_in_both_labels": sum(bool(normal and attack) for normal, attack in counts.values()),
            "rows_in_label_pure_values": sum(
                normal + attack for normal, attack in counts.values() if not normal or not attack
            ),
        }
        for name, counts in identity_counts.items()
    }
    return {
        "rows": rows,
        "columns": len(columns),
        "predictors": len(columns) - 2,
        "column_names": columns,
        "dtypes": dtypes,
        "label_counts": dict(label_counts),
        "attack_counts": dict(attack_counts),
        "label_attack_mismatches": label_attack_mismatches,
        "identity_label_overlap": identity_summary,
        "missing_values": dict(missing),
        "infinite_values": dict(infinite),
        "duplicate_rows": duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="CSV or single-CSV ZIP")
    parser.add_argument("--output", default="artifacts/security/nf_unsw_nb15_v2_audit.json")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    report = audit_chunks(pd.read_csv(args.input, chunksize=args.chunk_size, low_memory=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
