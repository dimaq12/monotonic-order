import unittest

import numpy as np

import ideal_order
from ideal_order import IdealOrder


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260711)

    def assert_same_float_order(self, actual, expected):
        self.assertTrue(np.array_equal(np.isnan(actual), np.isnan(expected)))
        mask = ~np.isnan(expected)
        self.assertTrue(np.array_equal(actual[mask], expected[mask]))

    def test_public_api_and_version(self):
        self.assertEqual(ideal_order.__version__, "0.1.0")
        self.assertIs(ideal_order.sort, IdealOrder.sort)

    def test_compact_model_contract(self):
        training = self.rng.normal(size=20_001)
        with IdealOrder(training, n_bins=256) as model:
            self.assertLess(model.storage_bytes, 5000)
            self.assertAlmostEqual(model.median, np.median(training), places=14)
            self.assertAlmostEqual(model.q1, np.quantile(training, 0.25), places=14)
            self.assertAlmostEqual(model.q3, np.quantile(training, 0.75), places=14)
            self.assertLess(abs(model.rank(0.0)-np.mean(training < 0.0)), 1/256)
        with self.assertRaises(RuntimeError):
            _ = model.n

    def test_exact_sort_distributions_and_ieee_edges(self):
        arrays = [
            self.rng.normal(size=50_000),
            self.rng.uniform(-1, 1, size=50_000),
            np.repeat([7.0, 3.0, 9.0, 1.0], 10_000),
            np.array([np.nan, 0.0, -0.0, np.inf, -np.inf, -1.0, 1.0, np.nan]),
        ]
        for values in arrays:
            self.assert_same_float_order(ideal_order.sort(values),
                                         np.sort(values, kind="stable"))

    def test_other_exact_operations(self):
        values = np.array([3.0, 1.0, 3.0, 2.0, -0.0, 0.0])
        self.assertTrue(ideal_order.is_sorted(ideal_order.sort(values)))
        self.assertTrue(np.array_equal(ideal_order.top_k(values, 2), [3.0, 3.0]))
        self.assertTrue(np.array_equal(ideal_order.bottom_k(values, 2), [-0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
