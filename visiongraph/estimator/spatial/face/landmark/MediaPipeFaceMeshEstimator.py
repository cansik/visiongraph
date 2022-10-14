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
        super().__init__(min_score)

        self.detector: Optional[_mp_face_mesh.FaceMesh] = None

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_tracking_confidence = min_tracking_confidence

    def setup(self):
        self.detector = _mp_face_mesh.FaceMesh(static_image_mode=self.static_image_mode,
                                               min_detection_confidence=self.min_score,
                                               max_num_faces=self.max_num_faces,
                                               refine_landmarks=self.refine_landmarks,
                                               min_tracking_confidence=self.min_tracking_confidence)

    def process(self, image: np.ndarray) -> ResultList[BlazeFaceMesh]:
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
        self.detector.close()

    def configure(self, args: Namespace):
        super().configure(args)
