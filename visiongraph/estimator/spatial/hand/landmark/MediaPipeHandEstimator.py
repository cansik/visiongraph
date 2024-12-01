from argparse import Namespace
from enum import Enum
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.estimator.spatial.hand.landmark.HandLandmarkEstimator import HandLandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.hand.BlazeHand import BlazeHand
from visiongraph.result.spatial.hand.Handedness import Handedness
from visiongraph.util.VectorUtils import list_of_vector4D


class HandModelComplexity(Enum):
    """
    Enum to represent the complexity of the hand model.
    """
    Light = 0
    Normal = 1


_mp_hands = mp.solutions.hands


class MediaPipeHandEstimator(HandLandmarkEstimator[BlazeHand]):
    """
    A class to estimate hands using MediaPipe hand tracking.
    """

    def __init__(self, complexity: HandModelComplexity = HandModelComplexity.Normal,
                 min_score: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2):
        """
        Initializes the MediaPipe hand estimator.

        :param complexity: The model complexity.
        :param min_score: The minimum detection confidence.
        :param min_tracking_confidence: The minimum tracking confidence.
        :param static_image_mode: Whether the image mode is static.
        :param max_num_hands: The maximum number of hands to track.
        """
        super().__init__(min_score)

        self.static_image_mode = static_image_mode
        self.min_tracking_confidence = min_tracking_confidence
        self.complexity = complexity
        self.max_num_hands = max_num_hands
        self.detector: Optional[_mp_hands.Hands] = None

    def setup(self):
        """
        Initializes the MediaPipe hand detector.
        """
        self.detector = _mp_hands.Hands(static_image_mode=self.static_image_mode,
                                        model_complexity=self.complexity.value,
                                        min_detection_confidence=self.min_score,
                                        min_tracking_confidence=self.min_tracking_confidence,
                                        max_num_hands=self.max_num_hands)

    def process(self, image: np.ndarray, **kwargs) -> ResultList[BlazeHand]:
        """
        Processes an image and returns a list of detected hands.

        :param image: The input image.
        :param **kwargs: Additional keyword arguments.

        :return: A list of detected BlazeHands.
        """
        # pre-process image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detector.process(image)

        # check if results are there
        if not (results.multi_hand_landmarks and results.multi_handedness):
            return ResultList()

        raw_hands = zip(results.multi_hand_landmarks, results.multi_handedness)

        hands: ResultList[BlazeHand] = ResultList()
        for landmarks, handedness in raw_hands:
            # Convert landmarks to 3D vectors
            landmarks = [(lm.x, lm.y, lm.z, 1.0) for lm in landmarks.landmark]
            class_res = handedness.classification[0]
            handedness = Handedness.LEFT if class_res.label == "Left" else Handedness.RIGHT
            hands.append(BlazeHand(class_res.score, list_of_vector4D(landmarks), handedness))

        return hands

    def release(self):
        """
        Releases the MediaPipe hand detector.
        """
        self.detector.close()

    def configure(self, args: Namespace):
        """
        Configures the estimator based on the provided arguments.

        :param args: The configuration namespace.
        """
        super().configure(args)

        # todo: implement arg parse
        # self.model = args.face_model
        # self.min_score = args.min_detection_confidence_face
