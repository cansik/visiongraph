import unittest

from visiongraph.model.NMSOptions import NMSBatchMode, NMSOptions
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.LandmarkDetectionResult import LandmarkDetectionResult
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.ResultUtils import non_maximum_suppression, non_maximum_suppression_from_options
from visiongraph.util.VectorUtils import list_of_vector4D


def _detection(class_id: int, score: float, x: float, y: float, width: float, height: float):
    return ObjectDetectionResult(class_id, f"class-{class_id}", score, BoundingBox2D(x, y, width, height))


class ResultUtilsTests(unittest.TestCase):
    def test_non_maximum_suppression_keeps_highest_scoring_overlap(self):
        results = [
            _detection(0, 0.95, 0.10, 0.10, 0.40, 0.40),
            _detection(0, 0.70, 0.12, 0.12, 0.40, 0.40),
            _detection(0, 0.80, 0.70, 0.70, 0.20, 0.20),
        ]

        filtered = non_maximum_suppression(results, score_threshold=0.0, nms_threshold=0.3)

        self.assertEqual(2, len(filtered))
        self.assertEqual([0.95, 0.80], [result.score for result in filtered])

    def test_non_maximum_suppression_auto_batches_when_classes_differ(self):
        results = [
            _detection(0, 0.90, 0.10, 0.10, 0.40, 0.40),
            _detection(1, 0.85, 0.10, 0.10, 0.40, 0.40),
        ]
        options = NMSOptions(score_threshold=0.0, nms_threshold=0.3, batch_mode=NMSBatchMode.Auto)

        filtered = non_maximum_suppression_from_options(results, options)

        self.assertEqual(2, len(filtered))
        self.assertEqual([0, 1], [result.class_id for result in filtered])

    def test_object_detection_map_coordinates_respects_source_and_destination_rois(self):
        detection = _detection(0, 1.0, 0.20, 0.20, 0.10, 0.20)

        detection.map_coordinates(
            src_size=(100.0, 50.0),
            dest_size=(200.0, 200.0),
            src_roi=BoundingBox2D(10.0, 5.0, 50.0, 20.0),
            dest_roi=BoundingBox2D(20.0, 40.0, 100.0, 80.0),
        )

        self.assertAlmostEqual(0.20, detection.bounding_box.x_min)
        self.assertAlmostEqual(0.30, detection.bounding_box.y_min)
        self.assertAlmostEqual(0.10, detection.bounding_box.width)
        self.assertAlmostEqual(0.20, detection.bounding_box.height)

    def test_landmark_detection_map_coordinates_moves_landmarks_and_bbox(self):
        landmarks = list_of_vector4D(
            [
                (0.20, 0.20, 0.0, 1.0),
                (0.30, 0.40, 0.0, 1.0),
            ]
        )
        detection = LandmarkDetectionResult(0, "pose", 1.0, landmarks, BoundingBox2D(0.20, 0.20, 0.10, 0.20))

        detection.map_coordinates(
            src_size=(100.0, 50.0),
            dest_size=(200.0, 200.0),
            src_roi=BoundingBox2D(10.0, 5.0, 50.0, 20.0),
            dest_roi=BoundingBox2D(20.0, 40.0, 100.0, 80.0),
        )

        self.assertAlmostEqual(0.20, detection.bounding_box.x_min)
        self.assertAlmostEqual(0.30, detection.bounding_box.y_min)
        self.assertAlmostEqual(0.10, detection.bounding_box.width)
        self.assertAlmostEqual(0.20, detection.bounding_box.height)
        self.assertAlmostEqual(0.20, float(detection.landmarks.x[0]))
        self.assertAlmostEqual(0.30, float(detection.landmarks.y[0]))
        self.assertAlmostEqual(0.30, float(detection.landmarks.x[1]))
        self.assertAlmostEqual(0.50, float(detection.landmarks.y[1]))


if __name__ == "__main__":
    unittest.main()
