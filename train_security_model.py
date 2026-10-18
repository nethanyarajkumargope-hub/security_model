"""Train and evaluate a binary UNSW-NB15 security-risk model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DROP_COLUMNS = ("id", "attack_cat", "label")
THRESHOLD = 0.5
MIN_RECALL = 0.95


def deduplicate_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated labeled records while retaining the original dataset untouched."""
    return frame.drop_duplicates(subset=[column for column in frame.columns if column != "id"]).copy()


def features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(DROP_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required UNSW-NB15 columns: {sorted(missing)}")
    return frame.drop(columns=list(DROP_COLUMNS)).replace([np.inf, -np.inf], np.nan)


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    categorical = list(frame.select_dtypes(include=["object", "string", "category", "bool"]).columns)
    numeric = [column for column in frame.columns if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def candidates(frame: pd.DataFrame, seed: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor(frame)),
            ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor(frame)),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "random_forest_fine": Pipeline([
            ("preprocessor", build_preprocessor(frame)),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "extra_trees": Pipeline([
            ("preprocessor", build_preprocessor(frame)),
            ("classifier", ExtraTreesClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = THRESHOLD) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive_rate, recall, thresholds = roc_curve(labels, probabilities)
    eligible = np.flatnonzero((recall >= MIN_RECALL) & np.isfinite(thresholds))
    if not len(eligible):
        raise ValueError(f"No threshold satisfies minimum recall {MIN_RECALL}")
    balanced_accuracy = (recall + 1 - false_positive_rate) / 2
    return float(thresholds[eligible[np.argmax(balanced_accuracy[eligible])]])


def error_breakdown(frame: pd.DataFrame, labels: np.ndarray,
                    probabilities: np.ndarray, threshold: float = THRESHOLD) -> dict:
    analysis = frame.reset_index(drop=True).copy()
    analysis["actual_label"] = labels
    analysis["predicted_label"] = (probabilities >= threshold).astype(int)

    def grouped(actual_label: int, predicted_label: int) -> dict:
        population = analysis[analysis["actual_label"] == actual_label]
        errors = population[population["predicted_label"] == predicted_label]
        result = {"count": int(len(errors))}
        for column in ("proto", "service", "state", "attack_cat"):
            support = population.groupby(column, dropna=False).size()
            error_count = errors.groupby(column, dropna=False).size()
            rows = [{"value": str(value), "support": int(count),
                     "errors": int(error_count.get(value, 0)),
                     "error_rate": float(error_count.get(value, 0) / count)}
                    for value, count in support.items() if error_count.get(value, 0)]
            result[f"by_{column}"] = sorted(
                rows, key=lambda row: (row["errors"], row["error_rate"]), reverse=True
            )[:20]
        return result

    return {"false_positives": grouped(0, 1), "false_negatives": grouped(1, 0)}


def grouped_split(frame: pd.DataFrame, seed: int):
    x = features(frame)
    y = frame["label"].to_numpy(dtype=int)
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    fit_index, holdout_index = next(first.split(x, y, groups))
    second = GroupShuffleSplit(n_splits=1, test_size=2 / 3, random_state=seed + 1)
    validation_local, remainder_local = next(second.split(
        x.iloc[holdout_index], y[holdout_index], groups[holdout_index]
    ))
    remainder_index = holdout_index[remainder_local]
    third = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 2)
    calibration_local, threshold_local = next(third.split(
        x.iloc[remainder_index], y[remainder_index], groups[remainder_index]
    ))
    return (fit_index, holdout_index[validation_local], remainder_index[calibration_local],
            remainder_index[threshold_local])


def top_features(pipeline: Pipeline, limit: int = 20) -> list[dict]:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    else:
        importance = np.abs(classifier.coef_[0])
    order = np.argsort(importance)[::-1][:limit]
    return [{"feature": str(names[index]), "importance": float(importance[index])} for index in order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/unsw_nb15/UNSW_NB15_training-set.csv")
    parser.add_argument("--test", default="data/unsw_nb15/UNSW_NB15_testing-set.csv")
    parser.add_argument("--output", default="artifacts/security")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)
    if list(train.columns) != list(test.columns):
        raise ValueError("Training and test schemas differ")
    original_train_rows = len(train)
    if args.deduplicate:
        train = deduplicate_training_rows(train)

    fit_index, validation_index, calibration_index, threshold_index = grouped_split(train, args.seed)
    x, y = features(train), train["label"].to_numpy(dtype=int)
    validation_results = {}
    fitted_candidates = {}
    for name, model in candidates(x, args.seed).items():
        model.fit(x.iloc[fit_index], y[fit_index])
        probability = model.predict_proba(x.iloc[validation_index])[:, 1]
        validation_results[name] = metrics(y[validation_index], probability)
        fitted_candidates[name] = model

    selected_name = max(validation_results, key=lambda name: (
        validation_results[name]["balanced_accuracy"], validation_results[name]["f1"]
    ))
    selected_validation_probability = fitted_candidates[selected_name].predict_proba(
        x.iloc[validation_index]
    )[:, 1]
    validation_errors = error_breakdown(
        train.iloc[validation_index], y[validation_index], selected_validation_probability
    )
    development_index = np.concatenate([fit_index, validation_index])
    final_pipeline = clone(fitted_candidates[selected_name])
    final_pipeline.fit(x.iloc[development_index], y[development_index])
    calibrated = CalibratedClassifierCV(FrozenEstimator(final_pipeline), method="sigmoid")
    calibrated.fit(x.iloc[calibration_index], y[calibration_index])

    calibration_probability = calibrated.predict_proba(x.iloc[calibration_index])[:, 1]
    threshold_probability = calibrated.predict_proba(x.iloc[threshold_index])[:, 1]
    selected_threshold = select_threshold(y[threshold_index], threshold_probability)
    test_x, test_y = features(test), test["label"].to_numpy(dtype=int)
    test_probability = calibrated.predict_proba(test_x)[:, 1]
    test_prediction = (test_probability >= selected_threshold).astype(int)

    train_hashes = set(pd.util.hash_pandas_object(x, index=False).to_numpy())
    test_hashes = pd.util.hash_pandas_object(test_x, index=False)
    category_report = {}
    for category, indices in test.groupby("attack_cat").groups.items():
        positions = np.asarray(list(indices), dtype=int)
        category_report[str(category)] = {
            "support": int(len(positions)),
            "mean_risk": float(test_probability[positions].mean()),
            "predicted_attack_rate": float(test_prediction[positions].mean()),
        }

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "UNSW-NB15 binary classification: normal=0, attack=1",
        "selected_model": selected_name,
        "threshold": selected_threshold,
        "minimum_threshold_tuning_recall": MIN_RECALL,
        "versions": {"python_packages": {"scikit_learn": sklearn.__version__,
                                            "pandas": pd.__version__, "numpy": np.__version__}},
        "rows": {"official_train": int(len(train)), "fit": int(len(fit_index)),
                 "validation": int(len(validation_index)), "calibration": int(len(calibration_index)),
                 "threshold_tuning": int(len(threshold_index)), "official_test": int(len(test))},
        "training_data": {"deduplicated": args.deduplicate,
                          "original_rows": int(original_train_rows),
                          "rows_removed": int(original_train_rows - len(train))},
        "predictor_count": int(x.shape[1]),
        "dropped_columns": list(DROP_COLUMNS),
        "validation_candidates": validation_results,
        "validation_errors": validation_errors,
        "calibration": metrics(y[calibration_index], calibration_probability, selected_threshold),
        "threshold_tuning": metrics(y[threshold_index], threshold_probability, selected_threshold),
        "official_test": metrics(test_y, test_probability, selected_threshold),
        "official_test_feature_overlap_with_train": {
            "rows": int(test_hashes.isin(train_hashes).sum()),
            "fraction": float(test_hashes.isin(train_hashes).mean()),
        },
        "attack_categories": category_report,
        "top_features": top_features(final_pipeline),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": calibrated, "threshold": selected_threshold,
                 "feature_columns": list(x.columns), "selected_model": selected_name},
                output / "security_model.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"id": test["id"], "attack_cat": test["attack_cat"],
                  "actual_label": test_y, "attack_probability": test_probability,
                  "predicted_label": test_prediction}).to_csv(output / "test_predictions.csv", index=False)
    print(json.dumps({"selected_model": selected_name, "official_test": report["official_test"],
                      "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
