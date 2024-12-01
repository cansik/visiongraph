from argparse import Namespace
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.estimator.spatial.face.landmark.FaceLandmarkEstimator import FaceLandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFaceMesh import BlazeFaceMesh
from visiongraph.util.VectorUtils import list_of_vector4D

_mp_face_mesh = mp.solutions.face_mesh


class MediaPipeFaceMeshEstimator(FaceLandmarkEstimator[BlazeFaceMesh]):

    def __init__(self, static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_score: float = 0.5,
                 min_tracking_confidence=0.5):
        """
        Initializes a MediaPipe FaceMeshEstimator.

        :param static_image_mode: Whether to use the static image mode.
        :param max_num_faces: The maximum number of faces to detect.
        :param refine_landmarks: Whether to refine the landmarks.
        :param min_score: The minimum detection confidence score.
        :param min_tracking_confidence: The minimum tracking confidence.
        """
        super().__init__(min_score)

        self.detector: Optional[_mp_face_mesh.FaceMesh] = None

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_tracking_confidence = min_tracking_confidence

    def setup(self):
        """
        Sets up the MediaPipe FaceMesh detector.
        """
        self.detector = _mp_face_mesh.FaceMesh(static_image_mode=self.static_image_mode,
                                               min_detection_confidence=self.min_score,
                                               max_num_faces=self.max_num_faces,
                                               refine_landmarks=self.refine_landmarks,
                                               min_tracking_confidence=self.min_tracking_confidence)

    def process(self, image: np.ndarray) -> ResultList[BlazeFaceMesh]:
        """
        Processes an image to detect faces and estimate landmarks.

        :param image: The input image.

        :return: A list of detected faces with estimated landmarks.
        """
        # pre-process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detector.process(image)

        # check if results are there
        if not results.multi_face_landmarks:
            return ResultList()

        faces: ResultList[BlazeFaceMesh] = ResultList()

        for face_landmarks in results.multi_face_landmarks:
            relative_key_points = face_landmarks.landmark

            landmarks = [(rkp.x, rkp.y, rkp.z, 1.0) for rkp in relative_key_points]
            faces.append(BlazeFaceMesh(1.0, list_of_vector4D(landmarks)))

        return faces

    def release(self):
        """
        Releases the MediaPipe FaceMesh detector.
        """
        self.detector.close()

    def configure(self, args: Namespace):
        """
        Configures the estimator based on the provided arguments.

        :param args: The configuration arguments.
        """
        super().configure(args)
