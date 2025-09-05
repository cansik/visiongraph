import unittest

import numpy as np
import vector

from visiongraph.dsp.OneEuroFilterNumpy import OneEuroFilterNumpy
from visiongraph.dsp.VectorNumpySmoothFilter import VectorNumpySmoothFilter


class DSPTest(unittest.TestCase):
    def test_numpy_vector_filter(self):
        data1 = vector.array([
            (1.1, 2.1), (1.2, 2.2), (1.3, 2.3), (1.4, 2.4), (1.5, 2.5)
        ], dtype=[("x", float), ("y", float)])

        data2 = vector.array([
            (1.2, 2.2), (1.1, 2.1), (1.5, 2.5), (1.4, 2.4), (1.5, 2.5)
        ], dtype=[("x", float), ("y", float)])

        np_filter = VectorNumpySmoothFilter(OneEuroFilterNumpy(np.full((5, 5), 1.0)))

        # first pass (re-init)
        np_filter.process(data1)

        # second pass (real filtering)
        np_filter.process(data2)


if __name__ == '__main__':
    unittest.main()
