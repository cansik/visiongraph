import unittest

from visiongraph.util.MathUtils import intersection_over_union


class MathUtilsTests(unittest.TestCase):
    def test_intersection_over_union_returns_exact_one_for_identical_boxes(self):
        iou = intersection_over_union([0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0])

        self.assertEqual(1.0, iou)

    def test_intersection_over_union_returns_zero_for_zero_union(self):
        iou = intersection_over_union([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])

        self.assertEqual(0.0, iou)


if __name__ == "__main__":
    unittest.main()
