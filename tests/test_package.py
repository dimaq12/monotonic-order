import unittest
import uuid

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
        self.assertEqual(ideal_order.__version__, "0.3.0")
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

    def test_radix_argsort_uint64_and_int64_is_stable(self):
        for values in (
            np.array([3, 1, 3, 2, 1, 0], dtype=np.uint64),
            np.array([0, -1, np.iinfo(np.int64).min, 7, -1,
                      np.iinfo(np.int64).max], dtype=np.int64),
        ):
            expected = np.argsort(values, kind="stable")
            actual = ideal_order.radix_argsort(values)
            self.assertTrue(np.array_equal(actual, expected))
            expected_desc = np.asarray(sorted(range(len(values)),
                                              key=lambda i: int(values[i]), reverse=True))
            self.assertTrue(np.array_equal(ideal_order.radix_argsort(values, descending=True),
                                           expected_desc))

    def test_radix_argsort_float_total_order_and_null_policy(self):
        values = np.array([np.nan, 0.0, -0.0, np.inf, -np.inf, 2.0, -3.0,
                           np.nan, 2.0])
        ordered = values[ideal_order.radix_argsort(values)]
        self.assertTrue(np.all(np.isnan(ordered[-2:])))
        self.assertTrue(np.array_equal(ordered[:-2], [-np.inf, -3.0, -0.0, 0.0,
                                                       2.0, 2.0, np.inf]))
        zero_signs = np.signbit(ordered[np.equal(ordered, 0.0)])
        self.assertTrue(np.array_equal(zero_signs, [True, False]))
        first = values[ideal_order.radix_argsort(values, nulls="first")]
        self.assertTrue(np.all(np.isnan(first[:2])))
        descending = values[ideal_order.radix_argsort(values, descending=True)]
        self.assertTrue(np.array_equal(descending[:-2], [np.inf, 2.0, 2.0, 0.0,
                                                          -0.0, -3.0, -np.inf]))
        with self.assertRaises(ValueError):
            ideal_order.radix_argsort(values, nulls="error")

    def test_apply_order_and_arbitrary_payload(self):
        payload = ["three-a", "one-a", "three-b", "two", "one-b"]
        keys = np.array([3, 1, 3, 2, 1], dtype=np.int64)
        permutation = ideal_order.radix_argsort(keys)
        expected = ["one-a", "one-b", "two", "three-a", "three-b"]
        self.assertEqual(ideal_order.apply_order(payload, permutation), expected)
        self.assertEqual(ideal_order.order_by(payload, keys=keys), expected)

        records = [{"name": "c", "score": 2}, {"name": "a", "score": 1},
                   {"name": "b", "score": 1}]
        result = ideal_order.order_by(records, key=lambda row: np.int64(row["score"]))
        self.assertEqual([row["name"] for row in result], ["a", "b", "c"])

    def test_invalid_argsort_and_permutation_inputs(self):
        with self.assertRaises(TypeError):
            ideal_order.radix_argsort(np.array([1, 2], dtype=np.int32))
        with self.assertRaises(ValueError):
            ideal_order.radix_argsort(np.ones((2, 2), dtype=np.uint64))
        with self.assertRaises(ValueError):
            ideal_order.apply_order([1, 2], [0, 0])

    def test_randomized_argsort_properties(self):
        for dtype in (np.uint64, np.int64, np.float64):
            for size in (0, 1, 2, 17, 1000):
                if dtype is np.uint64:
                    values = self.rng.integers(0, 31, size=size, dtype=np.uint64)
                elif dtype is np.int64:
                    values = self.rng.integers(-15, 16, size=size, dtype=np.int64)
                else:
                    values = self.rng.integers(-15, 16, size=size).astype(np.float64)
                permutation = ideal_order.radix_argsort(values)
                self.assertTrue(np.array_equal(np.sort(permutation), np.arange(size)))
                self.assertTrue(np.array_equal(permutation,
                                               np.argsort(values, kind="stable")))

    def test_apply_order_numpy_axis(self):
        payload = np.array([[30, 10, 20], [3, 1, 2]])
        permutation = ideal_order.radix_argsort(np.array([3, 1, 2], dtype=np.int64))
        actual = ideal_order.apply_order(payload, permutation, axis=1)
        self.assertTrue(np.array_equal(actual, [[10, 20, 30], [1, 2, 3]]))

    def test_lexargsort_multiple_fields_and_directions(self):
        primary = np.array([1, 0, 1, 0, 1], dtype=np.int64)
        secondary = np.array([2, 3, 1, 3, 1], dtype=np.int64)
        actual = ideal_order.radix_lexargsort(primary, secondary)
        self.assertTrue(np.array_equal(actual, [1, 3, 2, 4, 0]))
        mixed = ideal_order.radix_lexargsort(primary, secondary,
                                             descending=(False, True))
        expected = np.asarray(sorted(range(5),
                                     key=lambda i: (int(primary[i]), -int(secondary[i]))))
        self.assertTrue(np.array_equal(mixed, expected))

    def test_tuple_key_orders_records_lexicographically(self):
        records = [
            {"group": 1, "score": 2, "name": "c"},
            {"group": 0, "score": 3, "name": "a"},
            {"group": 1, "score": 1, "name": "b"},
            {"group": 0, "score": 3, "name": "d"},
        ]
        result = ideal_order.order_by(
            records,
            key=lambda row: (np.int64(row["group"]), np.int64(row["score"])),
        )
        self.assertEqual([row["name"] for row in result], ["a", "d", "b", "c"])

    def test_datetime_and_nat_policies(self):
        dates = np.array(["2025-01-02", "NaT", "2024-01-01", "2025-01-01"],
                         dtype="datetime64[D]")
        ordered = dates[ideal_order.radix_argsort(dates)]
        self.assertTrue(np.array_equal(ordered[:-1], np.array(
            ["2024-01-01", "2025-01-01", "2025-01-02"], dtype="datetime64[D]")))
        self.assertTrue(np.isnat(ordered[-1]))
        first = dates[ideal_order.radix_argsort(dates, descending=True, nulls="first")]
        self.assertTrue(np.isnat(first[0]))
        self.assertTrue(np.array_equal(first[1:], np.array(
            ["2025-01-02", "2025-01-01", "2024-01-01"], dtype="datetime64[D]")))

    def test_uuid_uses_full_128_bit_integer_order(self):
        values = [uuid.UUID(int=2**80), uuid.UUID(int=3), uuid.UUID(int=2**80),
                  uuid.UUID(int=1)]
        actual = ideal_order.radix_argsort(values)
        expected = np.asarray(sorted(range(len(values)), key=lambda i: values[i].int))
        self.assertTrue(np.array_equal(actual, expected))
        descending = ideal_order.radix_argsort(values, descending=True)
        expected_desc = np.asarray(sorted(range(len(values)),
                                          key=lambda i: values[i].int, reverse=True))
        self.assertTrue(np.array_equal(descending, expected_desc))

    def test_uuid_none_null_policy(self):
        values = [uuid.UUID(int=2), None, uuid.UUID(int=1), None]
        last = ideal_order.apply_order(values, ideal_order.radix_argsort(values))
        self.assertEqual(last[:2], [uuid.UUID(int=1), uuid.UUID(int=2)])
        self.assertEqual(last[2:], [None, None])
        first = ideal_order.apply_order(values,
                                        ideal_order.radix_argsort(values, nulls="first"))
        self.assertEqual(first[:2], [None, None])


if __name__ == "__main__":
    unittest.main()
