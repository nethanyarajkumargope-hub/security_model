"""Train a leakage-resistant binary model for NF-UNSW-NB15-v2."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from train_security_model import MIN_RECALL, metrics, select_threshold


LABEL_COLUMN = "Label"
ATTACK_COLUMN = "Attack"
IDENTITY_COLUMNS = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")
CATEGORICAL_CODE_COLUMNS = (
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
)


def grouped_folds(labels: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    fold = np.empty(len(labels), dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for number, (_, held_out) in enumerate(
        splitter.split(np.zeros((len(labels), 1), dtype=np.uint8), labels, groups)
    ):
        fold[held_out] = number
    return {
        "fit": np.flatnonzero(fold >= 4),
        "validation": np.flatnonzero(fold == 3),
        "calibration": np.flatnonzero(fold == 2),
        "threshold_tuning": np.flatnonzero(fold == 1),
        "held_out_test": np.flatnonzero(fold == 0),
    }


def build_model(feature_columns: list[str], seed: int) -> Pipeline:
    categorical = [column for column in CATEGORICAL_CODE_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    preprocessor = ColumnTransformer([
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                       dtype=np.float32), categorical),
        ("numeric", "passthrough", numeric),
    ])
    categorical_mask = [True] * len(categorical) + [False] * len(numeric)
    classifier = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, class_weight="balanced",
        categorical_features=categorical_mask, random_state=seed,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NF-UNSW-NB15-v2 CSV or single-CSV ZIP")
    parser.add_argument("--output", default="artifacts/netflow-security")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    header = pd.read_csv(args.input, nrows=0).columns.tolist()
    required = {LABEL_COLUMN, ATTACK_COLUMN, *IDENTITY_COLUMNS}
    missing = required - set(header)
    if missing:
        raise ValueError(f"NF-UNSW-NB15-v2 columns missing: {sorted(missing)}")
    dtypes = {column: np.float32 for column in header if column not in required}
    dtypes.update({LABEL_COLUMN: np.int8, ATTACK_COLUMN: "category",
                   IDENTITY_COLUMNS[0]: "category", IDENTITY_COLUMNS[1]: "category"})
    frame = pd.read_csv(args.input, dtype=dtypes, low_memory=False)
    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Label must contain both 0 and 1")

    groups = pd.util.hash_pandas_object(frame[list(IDENTITY_COLUMNS)], index=False).to_numpy()
    split = grouped_folds(labels, groups, args.seed)
    feature_columns = [column for column in header if column not in required]
    features = frame[feature_columns]
    model = build_model(feature_columns, args.seed)
    model.fit(features.iloc[split["fit"]], labels[split["fit"]])
    validation_risk = model.predict_proba(features.iloc[split["validation"]])[:, 1]

    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(features.iloc[split["calibration"]], labels[split["calibration"]])
    threshold_risk = calibrated.predict_proba(features.iloc[split["threshold_tuning"]])[:, 1]
    threshold = select_threshold(labels[split["threshold_tuning"]], threshold_risk)
    test_risk = calibrated.predict_proba(features.iloc[split["held_out_test"]])[:, 1]

    test_positions = split["held_out_test"]
    category_report = {}
    for category, positions in frame.iloc[test_positions].reset_index().groupby(
        ATTACK_COLUMN, observed=True
    ).groups.items():
        local = np.asarray(list(positions), dtype=int)
        category_report[str(category)] = {
            "support": int(len(local)),
            "mean_risk": float(test_risk[local].mean()),
            "predicted_attack_rate": float((test_risk[local] >= threshold).mean()),
        }

    split_report = {
        name: {"rows": int(len(indices)), "attacks": int(labels[indices].sum()),
               "attack_fraction": float(labels[indices].mean()),
               "ip_pairs": int(np.unique(groups[indices]).size)}
        for name, indices in split.items()
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "NF-UNSW-NB15-v2 binary classification: benign=0, attack=1",
        "selected_model": "hist_gradient_boosting",
        "threshold": threshold,
        "minimum_threshold_tuning_recall": MIN_RECALL,
        "versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
        "predictor_count": len(feature_columns),
        "dropped_columns": [*IDENTITY_COLUMNS, LABEL_COLUMN, ATTACK_COLUMN],
        "split_strategy": "10-fold stratified grouping by source-destination IP pair",
        "splits": split_report,
        "validation": metrics(labels[split["validation"]], validation_risk),
        "threshold_tuning": metrics(labels[split["threshold_tuning"]], threshold_risk, threshold),
        "held_out_test": metrics(labels[test_positions], test_risk, threshold),
        "held_out_attack_categories": category_report,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "estimator": calibrated,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "categorical_columns": [],
        "model_categorical_columns": list(CATEGORICAL_CODE_COLUMNS),
        "selected_model": "hist_gradient_boosting",
        "dataset": "NF-UNSW-NB15-v2",
        "label_column": LABEL_COLUMN,
        "attack_column": ATTACK_COLUMN,
    }
    joblib.dump(artifact, output / "security_model.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": report["selected_model"], "threshold": threshold,
                      "held_out_test": report["held_out_test"],
                      "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
