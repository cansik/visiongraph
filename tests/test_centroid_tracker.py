import unittest

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.tracker.CentroidTracker import CentroidTracker


def _detection(x: float, y: float, width: float = 1.0, height: float = 1.0) -> ObjectDetectionResult:
    return ObjectDetectionResult(0, "person", 1.0, BoundingBox2D(x, y, width, height))


class CentroidTrackerTests(unittest.TestCase):
    def test_tracker_keeps_id_across_one_missing_frame(self):
        tracker = CentroidTracker()
        tracker.max_lost = 1
        tracker.setup()

        first = tracker.process([_detection(0.0, 0.0)])
        tracking_id = first[0].tracking_id

        missing = tracker.process([])
        second = tracker.process([_detection(0.1, 0.0)])

        self.assertEqual(0, len(missing))
        self.assertEqual(1, len(second))
        self.assertEqual(tracking_id, second[0].tracking_id)

    def test_disabled_tracker_returns_detections_unchanged(self):
        tracker = CentroidTracker()
        tracker.enabled = False
        detections = [_detection(0.0, 0.0)]

        result = tracker.process(detections)

        self.assertIs(detections, result)
        self.assertEqual(-1, result[0].tracking_id)


if __name__ == "__main__":
    unittest.main()
