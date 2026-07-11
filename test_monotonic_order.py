import unittest

import numpy as np

from monotonic_order import MonotonicOrder


class MonotonicOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(20260711)
        cls.training = cls.rng.standard_normal(20_001)
        cls.order = MonotonicOrder(cls.training, n_bins=256)

    @classmethod
    def tearDownClass(cls):
        cls.order.close()

    def assertSameFloatOrder(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.array_equal(np.isnan(actual), np.isnan(expected)))
        mask = ~np.isnan(expected)
        self.assertTrue(np.array_equal(actual[mask], expected[mask]))

    def test_model_does_not_retain_training_array(self):
        self.assertFalse(any(isinstance(v, np.ndarray) for v in self.order.__dict__.values()))
        self.assertLess(self.order.storage_bytes, 5000)

    def test_exact_stored_statistics(self):
        self.assertEqual(self.order.min, self.training.min())
        self.assertEqual(self.order.max, self.training.max())
        self.assertAlmostEqual(self.order.q1, np.quantile(self.training, 0.25), places=14)
        self.assertAlmostEqual(self.order.median, np.quantile(self.training, 0.5), places=14)
        self.assertAlmostEqual(self.order.q3, np.quantile(self.training, 0.75), places=14)
        self.assertAlmostEqual(
            self.order.mad,
            np.quantile(np.abs(self.training - np.median(self.training)), 0.5),
            places=14,
        )

    def test_reference_rank_is_monotone_and_accurate(self):
        queries = np.linspace(self.order.min, self.order.max, 10_000)
        ranks = self.order.rank_array(queries)
        self.assertTrue(np.all(ranks[1:] >= ranks[:-1]))
        truth = np.searchsorted(np.sort(self.training), queries, side="left") / len(self.training)
        self.assertLess(np.max(np.abs(ranks - truth)), 1.0 / self.order.n_bins + 1e-12)

    def test_quantile_knots_are_exact(self):
        for j in range(self.order.n_bins + 1):
            q = j / self.order.n_bins
            self.assertAlmostEqual(self.order.quantile(q), np.quantile(self.training, q), places=13)

    def test_exact_sort_across_distributions(self):
        arrays = [
            self.rng.standard_normal(50_000),
            self.rng.uniform(-1, 1, 50_000),
            self.rng.exponential(size=50_000),
            np.repeat([7.0, 3.0, 9.0, 1.0], 10_000),
            np.zeros(50_000),
            np.arange(50_000.0)[::-1],
        ]
        for x in arrays:
            with self.subTest(size=x.size):
                self.assertSameFloatOrder(MonotonicOrder.sort(x), np.sort(x, kind="stable"))

    def test_ieee_edges_and_nan_policy(self):
        x = np.array([np.nan, 0.0, -0.0, np.inf, -np.inf, -1.0, 1.0, np.nan])
        got = MonotonicOrder.sort(x)
        expected = np.sort(x, kind="stable")
        self.assertSameFloatOrder(got, expected)
        self.assertTrue(MonotonicOrder.is_sorted(got))
        self.assertFalse(MonotonicOrder.is_sorted(x))

    def test_unique_and_k(self):
        x = np.array([3.0, 1.0, 3.0, 2.0, -0.0, 0.0])
        self.assertTrue(np.array_equal(MonotonicOrder.unique(x), np.array([-0.0, 0.0, 1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(MonotonicOrder.bottom_k(x, 2), np.sort(x, kind="stable")[:2]))
        self.assertTrue(np.array_equal(MonotonicOrder.top_k(x, 2), np.array([3.0, 3.0])))

    def test_ranges_and_reverse(self):
        x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        self.assertEqual(MonotonicOrder.count_between(x, 2.0, 4.0), 3)
        self.assertTrue(np.array_equal(MonotonicOrder.sort_reverse(x), np.array([5., 4., 3., 2., 1.])))

    def test_invalid_training(self):
        with self.assertRaises(ValueError):
            MonotonicOrder([])
        with self.assertRaises(ValueError):
            MonotonicOrder([1.0, np.nan])


if __name__ == "__main__":
    unittest.main()
