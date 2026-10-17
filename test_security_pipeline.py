import unittest

import pandas as pd

from train_security_model import DROP_COLUMNS, build_preprocessor, features


class SecurityPipelineTest(unittest.TestCase):
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
