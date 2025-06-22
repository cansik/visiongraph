from argparse import Namespace
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.face.landmark.FaceLandmarkEstimator import FaceLandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.result.spatial.face.BlendShape import BlendShape
from visiongraph.util.TimeUtils import HighPrecisionTimer
from visiongraph.util.VectorUtils import list_of_vector4D


class MediaPipeFaceMeshEstimator(FaceLandmarkEstimator[BlazeFaceMesh]):
    """
    A MediaPipe-based estimator for detecting facial landmarks and attributes.

    This class provides functionality to detect facial landmarks, blend shapes,
    and transformation matrices using MediaPipe's Face Landmarker.
    """

    def __init__(self, static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 min_face_detection_confidence: float = 0.5,
                 min_face_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 output_face_blendshapes: bool = False,
                 output_facial_transformation_matrixes: bool = False,
                 task: Asset = RepositoryAsset("face_landmarker.task")):
        """
        Initializes a MediaPipe FaceMeshEstimator.

        :param static_image_mode: Whether to use the static image mode.
        :param max_num_faces: The maximum number of faces to detect.
        :param min_face_detection_confidence: The minimum detection confidence score.
        :param min_face_presence_confidence: The minimum confidence for face presence detection.
        :param min_tracking_confidence: The minimum confidence for tracking landmarks.
        :param output_face_blendshapes: Whether to output face blend shapes.
        :param output_facial_transformation_matrixes: Whether to output transformation matrices.
        :param task: MediaPipe task to use for face mesh estimation.
        """
        super().__init__(min_face_detection_confidence)

        self.detector: Optional[vision.FaceLandmarker] = None

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.output_face_blendshapes = output_face_blendshapes
        self.output_facial_transformation_matrixes = output_facial_transformation_matrixes

        self.task = task

        self.timer = HighPrecisionTimer(ensure_monotonic=True)

    def setup(self):
        """
        Sets up the MediaPipe FaceMesh detector.

        Configures and initializes the MediaPipe FaceLandmarker using the specified
        task options and the running mode (image or video).
        """
        running_mode = VisionTaskRunningMode.IMAGE if self.static_image_mode else VisionTaskRunningMode.VIDEO

        # buffer loading because of https://github.com/google-ai-edge/mediapipe/issues/5343
        base_options = python.BaseOptions(model_asset_buffer=open(self.task.path, "rb").read())
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               running_mode=running_mode,
                                               min_face_detection_confidence=self.min_score,
                                               min_face_presence_confidence=self.min_face_presence_confidence,
                                               min_tracking_confidence=self.min_tracking_confidence,
                                               output_face_blendshapes=self.output_face_blendshapes,
                                               output_facial_transformation_matrixes=self.output_facial_transformation_matrixes,
                                               num_faces=self.max_num_faces)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, image: np.ndarray, timestamp_ms: Optional[int] = None) -> ResultList[BlazeFaceMesh]:
        """
        Processes an image to detect faces and estimate landmarks.

        :param image: The input image as a NumPy array in BGR format.
        :param timestamp_ms: The timestamp of the input video frame in milliseconds. If None, time.monotonic is used.

        :return: A ResultList of BlazeFaceMesh objects representing the detected faces
                 and their respective landmarks, blend shapes, and transformation matrices.
        """
        # Pre-process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Run inference
        if self.static_image_mode:
            results = self.detector.detect(input_frame)
        else:
            if timestamp_ms is None:
                timestamp_ms = int(self.timer.time_ms())

            results = self.detector.detect_for_video(input_frame, timestamp_ms=timestamp_ms)

        if len(results.face_landmarks) == 0:
            return ResultList()

        faces: ResultList[BlazeFaceMesh] = ResultList()
        for i, face_landmarks in enumerate(results.face_landmarks):
            landmarks = [(rkp.x, rkp.y, rkp.z, 1.0) for rkp in face_landmarks]
            face_mesh = BlazeFaceMesh(1.0, list_of_vector4D(landmarks))

            if self.output_face_blendshapes:
                blend_shapes = results.face_blendshapes[i]
                face_mesh.blend_shapes = [BlendShape(b.index, b.category_name, b.score) for b in blend_shapes]

            if self.output_facial_transformation_matrixes:
                face_mesh.transformation_matrix = results.facial_transformation_matrixes[i]

            faces.append(face_mesh)

        return faces

    def release(self):
        """
        Releases the MediaPipe FaceMesh detector.

        Closes any resources held by the FaceLandmarker to free up memory.
        """
        self.detector.close()

    def configure(self, args: Namespace):
        """
        Configures the estimator based on the provided arguments.

        :param args: The configuration arguments as a Namespace object.
        """
        super().configure(args)
