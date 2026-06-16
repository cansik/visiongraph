import unittest

import numpy as np

from visiongraph.dsp.OneEuroFilter import OneEuroFilter
from visiongraph.dsp.OneEuroFilterNumpy import OneEuroFilterNumpy


class OneEuroFilterTests(unittest.TestCase):
    def test_scalar_filter_smooths_step_input_monotonically(self):
        filter_ = OneEuroFilter(x0=0.0, t0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)

        first = filter_(1.0, t=1.0)
        second = filter_(1.0, t=2.0)

        self.assertGreater(first, 0.0)
        self.assertLess(first, 1.0)
        self.assertGreater(second, first)
        self.assertLessEqual(second, 1.0)

    def test_numpy_filter_preserves_previous_value_for_invalid_entries(self):
        filter_ = OneEuroFilterNumpy(
            np.array([1.0, 2.0], dtype=float),
            t0=0.0,
            min_cutoff=1.0,
            beta=0.0,
            d_cutoff=1.0,
            invalid_value=-1.0,
        )

        result = filter_(np.array([3.0, -1.0], dtype=float), t=1.0)

        self.assertGreater(result[0], 1.0)
        self.assertLess(result[0], 3.0)
        self.assertEqual(2.0, result[1])

    def test_numpy_filter_reinitializes_when_shape_changes(self):
        filter_ = OneEuroFilterNumpy(np.array([1.0, 2.0], dtype=float), t0=0.0)

        result = filter_(np.array([3.0, 4.0, 5.0], dtype=float), t=1.0)

        np.testing.assert_allclose(result, np.array([3.0, 4.0, 5.0], dtype=float))
        self.assertEqual((3,), filter_.data_shape)

    def test_scalar_and_numpy_filters_match_for_single_value(self):
        scalar_filter = OneEuroFilter(x0=0.0, t0=0.0, min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
        numpy_filter = OneEuroFilterNumpy(
            np.array([0.0], dtype=float),
            t0=0.0,
            min_cutoff=1.0,
            beta=0.5,
            d_cutoff=1.0,
        )

        scalar_value = scalar_filter(1.0, t=1.0)
        numpy_value = numpy_filter(np.array([1.0], dtype=float), t=1.0)

        self.assertAlmostEqual(scalar_value, float(numpy_value[0]))


if __name__ == "__main__":
    unittest.main()
