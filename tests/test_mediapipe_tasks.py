import unittest
from importlib.util import find_spec
from unittest.mock import Mock, patch

import cv2
import numpy as np

from visiongraph import vg


@unittest.skipUnless(find_spec("mediapipe"), "MediaPipe extra required")
class MediaPipeTasksTests(unittest.TestCase):
    def test_default_assets_use_the_visiongraph_mirror(self):
        from visiongraph.util.NetworkUtils import PUBLIC_DATA_URL

        for config in (
            vg.MediaPipeFaceDetectorConfig,
            vg.MediaPipeHandConfig,
            vg.MediaPipeHolisticConfig,
            vg.MediaPipeSelfieSegmentationConfig,
        ):
            for preset in config:
                with self.subTest(preset=preset):
                    self.assertEqual(PUBLIC_DATA_URL, preset.value.repository_url)

    def test_task_rebuild_preserves_lifecycle_and_timestamps(self):
        from mediapipe.tasks.python import vision

        original, replacement, reopened = Mock(), Mock(), Mock()
        create = Mock(side_effect=[original, ValueError("invalid options"), replacement, reopened])
        detect = Mock()
        detect_for_video = Mock()
        task = vg.MediaPipeTaskRunner(dict, create, detect, detect_for_video)
        self.addCleanup(task.release)
        asset = vg.LocalAsset("unused.task")
        frame = np.zeros((8, 16, 3), dtype=np.uint8)
        options = {"running_mode": vision.RunningMode.VIDEO, "min_detection_confidence": 0.5}

        task.setup(asset, **options)
        task.process(frame, asset, 10, **options)
        with patch("visiongraph.estimator.spatial.mediapipe.MediaPipeTaskRunner.monotonic_ns", return_value=10_000_000):
            task.process(frame, asset, **options)
        self.assertEqual(1, create.call_count)
        self.assertEqual([10, 11], [call.args[2] for call in detect_for_video.call_args_list])

        updated = options | {"min_detection_confidence": 0.8}
        with self.assertRaisesRegex(ValueError, "invalid options"):
            task.process(frame, asset, 12, **updated)
        original.close.assert_not_called()
        task.process(frame, asset, 12, **options)
        self.assertIs(original, detect_for_video.call_args.args[0])

        task.process(frame, asset, 13, **updated)
        original.close.assert_called_once()
        self.assertEqual(0.8, create.call_args.args[0]["min_detection_confidence"])
        self.assertIs(replacement, detect_for_video.call_args.args[0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            task.process(frame, asset, 13, **updated)

        task.release()
        task.release()
        replacement.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "Set up"):
            task.process(frame, asset, 0, **options)

        task.setup(asset, **options)
        task.process(frame, asset, 0, **options)
        self.assertIs(reopened, detect_for_video.call_args.args[0])
        detect.assert_not_called()

    def test_face_presets_return_normalized_faces(self):
        frame = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")
        for model in vg.MediaPipeFaceModel:
            with self.subTest(model=model), vg.MediaPipeFaceDetector(model, min_score=0.3) as detector:
                faces = detector.process(frame)
                self.assertEqual(1, len(faces))
                self.assertEqual(6, len(faces[0].landmarks))
                self.assertGreater(faces[0].bounding_box.width, 0)
                self.assertLessEqual(faces[0].bounding_box.width, 1)
                self.assertLessEqual(faces[0].bounding_box.height, 1)
                faces.annotate(frame.copy())

    def test_face_mesh_output_changes_on_the_next_video_frame(self):
        frame = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")
        with vg.MediaPipeFaceMeshEstimator() as estimator:
            self.assertIsNone(estimator.process(frame, 0)[0].blend_shapes)
            estimator.output_face_blendshapes = True
            estimator.output_facial_transformation_matrixes = True
            faces = estimator.process(frame, 1)
            self.assertEqual(1, len(faces))
            self.assertEqual(478, len(faces[0].landmarks))
            self.assertEqual(52, len(faces[0].blend_shapes))
            self.assertEqual((4, 4), faces[0].transformation_matrix.shape)
            faces.annotate(frame.copy())
            self.assertEqual([], estimator.process(np.zeros_like(frame), 2))

    def test_hand_landmarks_and_handedness(self):
        frame = cv2.imread("assets/hands-pexels-ketut-subiyanto-4126739.jpg")
        with vg.MediaPipeHandEstimator(static_image_mode=True) as estimator:
            hands = estimator.process(frame)
            self.assertEqual(2, len(hands))
            self.assertEqual({vg.Handedness.LEFT, vg.Handedness.RIGHT}, {hand.handedness for hand in hands})
            self.assertTrue(all(len(hand.landmarks) == 21 for hand in hands))
            hands.annotate(frame.copy())

    def test_pose_and_holistic_results(self):
        frame = cv2.imread("assets/pexels-pavel-danilyuk-6203560.jpg")
        with vg.MediaPipePoseEstimator.create(vg.MediaPipePoseConfig.Light, static_image_mode=True) as estimator:
            poses = estimator.process(frame)
            self.assertEqual(1, len(poses))
            self.assertEqual(33, len(poses[0].landmarks))
            self.assertEqual(frame.shape[:2], poses[0].mask.shape)
            self.assertEqual(np.uint8, poses[0].mask.dtype)
            poses.annotate(frame.copy())

        from visiongraph.estimator.spatial.pose import PoseEstimators

        self.assertEqual(
            {"mediapipe", "mediapipe-light", "mediapipe-heavy", "mediapipe-holistic"},
            {name for name in PoseEstimators if name.startswith("mediapipe")},
        )
        with PoseEstimators["mediapipe-holistic"]() as estimator:
            estimator.enable_segmentation = True
            poses = estimator.process(frame, 0)
            self.assertEqual(1, len(poses))
            self.assertEqual(33, len(poses[0].landmarks))
            self.assertEqual(frame.shape[:2], poses[0].segmentation_mask.shape)
            self.assertEqual(np.uint8, poses[0].segmentation_mask.dtype)
            poses.annotate(frame.copy())

    def test_selfie_presets_return_person_masks(self):
        frame = cv2.imread("assets/pexels-nguyendesigner-32123175.jpg")
        for model in vg.SelfieSegmentationModel:
            with self.subTest(model=model), vg.MediaPipeSelfieSegmentation.create(model) as estimator:
                results = estimator.process(frame)
                self.assertEqual(1, len(results))
                self.assertEqual(frame.shape[:2], results[0].mask.shape)
                self.assertEqual(np.uint8, results[0].mask.dtype)
                self.assertGreater(results[0].mask.max(), 200)
                self.assertLess(results[0].mask.min(), 50)
                results.annotate(frame.copy())
