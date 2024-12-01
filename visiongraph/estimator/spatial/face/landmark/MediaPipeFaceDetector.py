from argparse import Namespace
from enum import Enum
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.estimator.spatial.face.landmark.FaceLandmarkEstimator import FaceLandmarkEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.BlazeFace import BlazeFace
from visiongraph.util.VectorUtils import list_of_vector4D


class MediaPipeFaceModel(Enum):
    """
    Enum for MediaPipe face detection models.
    """

    Short_Range = 0
    Full_Range = 1


_mp_face_detection = mp.solutions.face_detection


class MediaPipeFaceDetector(FaceLandmarkEstimator[BlazeFace]):
    """
    A class to represent a MediaPipe face detector.
    """

    def __init__(self, model: MediaPipeFaceModel = MediaPipeFaceModel.Short_Range, min_score: float = 0.5):
        """
        Initializes the MediaPipeFaceDetector.

        :param model: The model used for face detection. Defaults to MediaPipeFaceModel.Short_Range.
        :param min_score: The minimum score for face detection. Defaults to 0.5.
        """

        super().__init__(min_score)

        self.detector: Optional[_mp_face_detection.FaceDetection] = None
        self.model = model

    def setup(self):
        """
        Sets up the FaceDetection object.

        """

        self.detector = _mp_face_detection.FaceDetection(model_selection=self.model.value,
                                                         min_detection_confidence=self.min_score)

    def process(self, image: np.ndarray) -> ResultList[BlazeFace]:
        """
        Processes an image for face detection.

        :param image: The input image.

        :return: A list of detected faces.
        """

        # pre-process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detector.process(image)

        # check if results are there
        if not results.detections:
            return ResultList()

        faces: ResultList[BlazeFace] = ResultList()

        for detection in results.detections:
            rbb = detection.location_data.relative_bounding_box
            relative_key_points = detection.location_data.relative_keypoints

            box = BoundingBox2D(rbb.xmin, rbb.ymin, rbb.width, rbb.height)
            landmarks = [(rkp.x, rkp.y, 0, 1.0) for rkp in relative_key_points]

            faces.append(BlazeFace(detection.score[0], list_of_vector4D(landmarks), box))

        return faces

    def release(self):
        """
        Releases the FaceDetection object.

        """

        self.detector.close()

    def configure(self, args: Namespace):
        """
        Configures the MediaPipeFaceDetector.

        :param args: The input arguments.

        """

        super().configure(args)

        # todo: implement arg parse
        # self.model = args.face_model
        # self.min_score = args.min_detection_confidence_face
