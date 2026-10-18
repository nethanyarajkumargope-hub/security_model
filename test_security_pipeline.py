import unittest

import numpy as np
import pandas as pd

from train_security_model import (
    DROP_COLUMNS,
    build_preprocessor,
    candidates,
    deduplicate_training_rows,
    error_breakdown,
    features,
    select_threshold,
)


class SecurityPipelineTest(unittest.TestCase):
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
