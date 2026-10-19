import unittest
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from audit_netflow_dataset import audit_chunks
from evaluate_security_model import evaluate_frame
from run_security_multiseed import summarize_runs
from security_risk import attack_risk, prepare_input, threshold_risk
from train_security_model import (
    DROP_COLUMNS,
    build_preprocessor,
    candidates,
    deduplicate_training_rows,
    error_breakdown,
    features,
    permutation_features,
    select_threshold,
)
from train_netflow_security_model import grouped_folds
from verify_security_release import build_manifest, load_verified_model, verify_manifest


class SecurityPipelineTest(unittest.TestCase):
    def test_netflow_folds_keep_ip_pairs_disjoint(self):
        groups = np.repeat(np.arange(20), 4)
        labels = np.repeat(np.arange(20) % 2, 4)
        split = grouped_folds(labels, groups, 42)
        self.assertEqual(sum(map(len, split.values())), len(labels))
        group_sets = [set(groups[index]) for index in split.values()]
        self.assertTrue(all(left.isdisjoint(right) for position, left in enumerate(group_sets)
                            for right in group_sets[position + 1:]))

    def test_netflow_audit_counts_labels_quality_and_duplicates(self):
        columns = {"IPV4_SRC_ADDR": ["a", "b"], "IPV4_DST_ADDR": ["x", "y"],
                   "feature": [1.0, 2.0], "Label": [0, 1], "Attack": ["Benign", "DoS"]}
        report = audit_chunks([pd.DataFrame(columns), pd.DataFrame(columns).iloc[[0]]])
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["label_counts"], {"0": 2, "1": 1})
        self.assertEqual(report["attack_counts"], {"Benign": 2, "DoS": 1})
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["label_attack_mismatches"], 0)
        self.assertEqual(report["identity_label_overlap"]["IP_PAIR"]["unique_values"], 2)

    def test_release_manifest_detects_model_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, metrics_path, manifest_path = (
                root / "model.joblib", root / "metrics.json", root / "manifest.json"
            )
            joblib.dump({"schema_version": 1, "selected_model": "test", "threshold": 0.5,
                         "feature_columns": ["dur"], "categorical_columns": []}, model_path)
            metrics_path.write_text(json.dumps({"official_test": {}, "versions": {}}), encoding="utf-8")
            manifest = build_manifest(model_path, metrics_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_manifest(model_path, manifest_path)["feature_count"], 1)
            self.assertEqual(load_verified_model(model_path, manifest_path)["selected_model"], "test")
            tampered = bytearray(model_path.read_bytes())
            tampered[-1] ^= 1
            model_path.write_bytes(tampered)
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                verify_manifest(model_path, manifest_path)
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                load_verified_model(model_path, manifest_path)

    def test_frozen_evaluation_requires_binary_labels_and_reports_categories(self):
        class Estimator:
            classes_ = np.array([0, 1])

            def predict_proba(self, frame):
                risk = np.array([0.1, 0.8, 0.4, 0.9])[:len(frame)]
                return np.column_stack([1 - risk, risk])

        artifact = {"estimator": Estimator(), "selected_model": "test", "threshold": 0.5,
                    "feature_columns": ["dur"], "categorical_columns": []}
        frame = pd.DataFrame({"dur": [1, 2, 3, 4], "label": [0, 1, 0, 1],
                              "attack_cat": ["Normal", "Generic", "Normal", "Generic"]})
        report = evaluate_frame(frame, artifact)
        self.assertEqual(report["metrics"]["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(report["attack_categories"]["Generic"]["support"], 2)
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            evaluate_frame(frame.assign(label=[0, 1, 2, 1]), artifact)
        with self.assertRaisesRegex(ValueError, "both label 0 and label 1"):
            evaluate_frame(frame.assign(label=0), artifact)

        netflow_artifact = {**artifact, "label_column": "Label", "attack_column": "Attack"}
        netflow_frame = frame.rename(columns={"label": "Label", "attack_cat": "Attack"})
        self.assertEqual(evaluate_frame(netflow_frame, netflow_artifact)["rows"], 4)

    def test_risk_adapter_uses_attack_class_and_saved_threshold(self):
        class ReversedEstimator:
            classes_ = np.array([1, 0])

            def predict_proba(self, frame):
                return np.array([[1.2, -0.2]] * len(frame))

        artifact = {"estimator": ReversedEstimator(), "threshold": 0.7,
                    "feature_columns": ["dur"], "categorical_columns": []}
        risk = attack_risk(pd.DataFrame({"dur": [0.1]}), artifact)
        np.testing.assert_array_equal(risk, [1.0])
        np.testing.assert_array_equal(threshold_risk(risk, artifact["threshold"]), [1])

        artifact["estimator"].classes_ = np.array([0])
        with self.assertRaisesRegex(ValueError, "attack class"):
            attack_risk(pd.DataFrame({"dur": [0.1]}), artifact)

    def test_prediction_input_enforces_saved_schema(self):
        artifact = {"feature_columns": ["dur", "proto"], "categorical_columns": ["proto"]}
        prepared = prepare_input(
            pd.DataFrame({"id": [1], "proto": ["tcp"], "dur": ["0.1"], "unused": [9]}),
            artifact,
        )
        self.assertEqual(prepared.columns.tolist(), ["dur", "proto"])
        self.assertEqual(prepared["dur"].dtype.kind, "f")
        with self.assertRaisesRegex(ValueError, "missing model features"):
            prepare_input(pd.DataFrame({"dur": [0.1]}), artifact)
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            prepare_input(pd.DataFrame({"dur": ["bad"], "proto": ["tcp"]}), artifact)
        duplicate = pd.DataFrame([[0.1, 0.2, "tcp"]], columns=["dur", "dur", "proto"])
        with self.assertRaisesRegex(ValueError, "duplicate columns"):
            prepare_input(duplicate, artifact)

    def test_multiseed_summary_reports_model_counts_and_ranges(self):
        runs = [
            {"selected_model": "a", "threshold": 0.4,
             "selected_validation": {"balanced_accuracy": 0.8},
             "threshold_tuning": {"recall": 0.95, "false_positive_rate": 0.1}},
            {"selected_model": "a", "threshold": 0.6,
             "selected_validation": {"balanced_accuracy": 0.9},
             "threshold_tuning": {"recall": 0.97, "false_positive_rate": 0.2}},
        ]
        summary = summarize_runs(runs)
        self.assertEqual(summary["selected_models"], {"a": 2})
        self.assertAlmostEqual(summary["threshold"]["mean"], 0.5)
        self.assertAlmostEqual(summary["validation_balanced_accuracy"]["maximum"], 0.9)

    def test_permutation_importance_reports_original_columns(self):
        frame = pd.DataFrame({
            "dur": [0.1, 0.2, 0.3, 1.0, 1.1, 1.2],
            "proto": ["udp", "udp", "udp", "tcp", "tcp", "tcp"],
        })
        labels = np.array([0, 0, 0, 1, 1, 1])
        model = candidates(frame, 42)["logistic_regression"].fit(frame, labels)
        importance = permutation_features(model, frame, labels, 42)
        self.assertEqual({row["feature"] for row in importance}, {"dur", "proto"})

    def test_error_breakdown_uses_class_support_for_rates(self):
        frame = pd.DataFrame({
            "proto": ["tcp", "tcp", "udp", "udp"],
            "service": ["http", "http", "dns", "dns"],
            "state": ["FIN", "FIN", "CON", "CON"],
            "attack_cat": ["Normal", "Normal", "Generic", "Generic"],
        })
        result = error_breakdown(
            frame, np.array([0, 0, 1, 1]), np.array([0.8, 0.2, 0.7, 0.3])
        )
        self.assertEqual(result["false_positives"]["count"], 1)
        self.assertEqual(result["false_negatives"]["count"], 1)
        self.assertEqual(result["false_positives"]["by_proto"][0]["error_rate"], 0.5)

    def test_expected_candidate_models_are_available(self):
        frame = pd.DataFrame({"dur": [0.1], "proto": ["tcp"]})
        self.assertEqual(
            set(candidates(frame, 42)),
            {"logistic_regression", "random_forest", "random_forest_fine", "extra_trees"},
        )

    def test_threshold_maximizes_balanced_accuracy_at_required_recall(self):
        threshold = select_threshold(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
        )
        self.assertAlmostEqual(threshold, 0.6)

    def test_deduplication_ignores_id_but_preserves_conflicting_labels(self):
        frame = pd.DataFrame({
            "id": [1, 2, 3], "dur": [0.1, 0.1, 0.1], "proto": ["tcp"] * 3,
            "attack_cat": ["Normal", "Normal", "Generic"], "label": [0, 0, 1],
        })
        cleaned = deduplicate_training_rows(frame)
        self.assertEqual(cleaned["id"].tolist(), [1, 3])

    def test_targets_are_removed_and_unknown_categories_are_supported(self):
        frame = pd.DataFrame({
            "id": [1, 2], "dur": [0.1, 0.2], "proto": ["tcp", "udp"],
            "attack_cat": ["Normal", "Generic"], "label": [0, 1],
        })
        x = features(frame)
        self.assertTrue(set(DROP_COLUMNS).isdisjoint(x.columns))
        preprocessor = build_preprocessor(x).fit(x)
        transformed = preprocessor.transform(pd.DataFrame({"dur": [0.3], "proto": ["new_protocol"]}))
        self.assertEqual(transformed.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
