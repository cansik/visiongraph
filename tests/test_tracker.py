import unittest

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.tracker.FlateTracker import FlateTracker
from visiongraph.tracker.ObjectAssignmentSolver import ObjectAssignmentSolver


def _detection(class_id: int, class_name: str, x: float, y: float, width: float = 1.0, height: float = 1.0):
    return ObjectDetectionResult(class_id, class_name, 1.0, BoundingBox2D(x, y, width, height))


class ObjectAssignmentSolverTests(unittest.TestCase):
    def test_solve_rejects_matches_above_max_cost(self):
        solver = ObjectAssignmentSolver(max_cost=0.5)
        source = _detection(0, "person", 0.0, 0.0)
        destination = _detection(0, "person", 10.0, 10.0)

        result = solver.solve([source], [destination])

        self.assertEqual({source: None}, result.assignments)
        self.assertEqual([destination], result.unassigned_destinations)
        self.assertEqual(0.5, result.costs[source])
        self.assertEqual([source], result.unassigned_sources)

    def test_iou_cost_function_honors_class_exclusive(self):
        track = _detection(0, "person", 0.0, 0.0, 2.0, 2.0)
        same_class = _detection(0, "person", 0.0, 0.0, 2.0, 2.0)
        different_class = _detection(1, "car", 0.0, 0.0, 2.0, 2.0)

        cost_matrix = ObjectAssignmentSolver.iou_cost_function(
            [track], [same_class, different_class], class_exclusive=True
        )

        self.assertEqual(0.0, cost_matrix[0, 0])
        self.assertGreater(cost_matrix[0, 1], 1e10)


class FlateTrackerTests(unittest.TestCase):
    def test_tracker_preserves_tracking_id_across_nearby_frames(self):
        tracker = FlateTracker(max_cost=5.0)
        tracker.setup()

        first_result = tracker.process([_detection(0, "person", 0.0, 0.0)])
        initial_tracking_id = first_result[0].tracking_id

        second_detection = _detection(0, "person", 0.2, 0.1)
        second_result = tracker.process([second_detection])

        self.assertEqual(1, len(second_result))
        self.assertEqual(initial_tracking_id, second_result[0].tracking_id)
        self.assertEqual(0, second_result[0].staleness)

    def test_tracker_drops_stale_tracks_after_max_lost(self):
        tracker = FlateTracker(max_lost=1)
        tracker.setup()
        tracker.process([_detection(0, "person", 0.0, 0.0)])

        first_missing = tracker.process([])
        second_missing = tracker.process([])

        self.assertEqual(0, len(first_missing))
        self.assertEqual(0, len(second_missing))

    def test_tracker_can_include_stale_tracks(self):
        tracker = FlateTracker(max_lost=2)
        tracker.include_stale = True
        tracker.setup()

        tracked = tracker.process([_detection(0, "person", 0.0, 0.0)])
        tracking_id = tracked[0].tracking_id

        stale_result = tracker.process([])

        self.assertEqual(1, len(stale_result))
        self.assertEqual(tracking_id, stale_result[0].tracking_id)
        self.assertEqual(1, stale_result[0].staleness)
        self.assertTrue(stale_result[0].is_stale)

    def test_class_aware_matching_prevents_cross_class_swaps(self):
        tracker = FlateTracker(max_cost=20.0, class_aware=True)
        tracker.setup()

        first_frame = tracker.process(
            [
                _detection(0, "person", 0.0, 0.0),
                _detection(1, "car", 10.0, 0.0),
            ]
        )
        first_ids = {item.class_name: item.tracking_id for item in first_frame}

        second_frame = tracker.process(
            [
                _detection(0, "person", 10.1, 0.0),
                _detection(1, "car", 0.1, 0.0),
            ]
        )
        second_ids = {item.class_name: item.tracking_id for item in second_frame}

        self.assertEqual(first_ids["person"], second_ids["person"])
        self.assertEqual(first_ids["car"], second_ids["car"])


if __name__ == "__main__":
    unittest.main()
