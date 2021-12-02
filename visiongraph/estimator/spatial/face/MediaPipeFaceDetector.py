from argparse import Namespace
from enum import Enum
from typing import List, Optional

import mediapipe as mp
import numpy as np

from visiongraph.estimator.spatial.face.landmark.FaceLandmarkEstimator import FaceLandmarkEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.face.BlazeFace import BlazeFace
from visiongraph.util.ResultUtils import list_of_vector4D


class MediaPipeFaceModel(Enum):
    Short_Range = 0
    Full_Range = 1


_mp_face_detection = mp.solutions.face_detection


class MediaPipeFaceDetector(FaceLandmarkEstimator):

    def __init__(self, model: MediaPipeFaceModel = MediaPipeFaceModel.Short_Range, min_score: float = 0.5):
        super().__init__(min_score)

        self.detector: Optional[_mp_face_detection.FaceDetection] = None
        self.model = model

    def setup(self):
        self.detector = _mp_face_detection.FaceDetection(model_selection=self.model.value,
                                                         min_detection_confidence=self.min_score)

    def estimate(self, image: np.ndarray, **kwargs) -> List[BlazeFace]:
        results = self.detector.process(image)

        # check if results are there
        if not results.detections:
            return []

        faces: List[BlazeFace] = []

        for detection in results.detections:
            rbb = detection.location_data.relative_bounding_box
            relative_key_points = detection.location_data.relative_keypoints

            box = BoundingBox2D(rbb.xmin, rbb.ymin, rbb.width, rbb.height)
            landmarks = [(rkp.x, rkp.y, 0, 0) for rkp in relative_key_points]

            faces.append(BlazeFace(detection.score[0], list_of_vector4D(landmarks), box))

        return faces

    def release(self):
        self.detector.close()

    def configure(self, args: Namespace):
        super().configure(args)

        # todo: implement arg parse
        # self.model = args.face_model
        # self.min_score = args.min_detection_confidence_face
