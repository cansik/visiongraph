from argparse import Namespace
from typing import Optional

import cv2
import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.hand.landmark.HandLandmarkEstimator import HandLandmarkEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.hand.Handedness import Handedness
from visiongraph.result.spatial.hand.OpenPoseHand import OpenPoseHand
from visiongraph.util.VectorUtils import list_of_vector4D

OPEN_POSE_KEYPOINT_COUNT = 21


class OpenPoseHandEstimator(HandLandmarkEstimator[OpenPoseHand]):

    def __init__(self, min_score: float = 0.5):
        super().__init__(min_score)

        self.model = RepositoryAsset("hand_pose_deploy.prototxt")
        self.weights = RepositoryAsset("hand_pose_iter_102000.caffemodel")
        self.input_size = 368
        self.resize_heatmaps: bool = False

        self.network: Optional[cv2.dnn_Net] = None

    def setup(self):
        self.network = cv2.dnn.readNetFromCaffe(self.model.path, self.weights.path)

    def process(self, image: np.ndarray, **kwargs) -> ResultList[OpenPoseHand]:
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (self.input_size, self.input_size), (0, 0, 0),
                                     swapRB=False, crop=False)

        self.network.setInput(blob)
        output = self.network.forward()

        hands: ResultList[OpenPoseHand] = ResultList()
        landmarks = []
        pose_score: float = 0.0

        for i in range(OPEN_POSE_KEYPOINT_COUNT):
            heatmap = output[0, i, :, :]

            if self.resize_heatmaps:
                heatmap = cv2.resize(heatmap, (self.input_size, self.input_size))

            h, w = heatmap.shape[:2]
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
            x = max_loc[0] / w
            y = max_loc[1] / h
            score = max_val

            pose_score += score
            landmarks.append((x, y, 0.0, score))

        pose_score = pose_score / OPEN_POSE_KEYPOINT_COUNT

        if pose_score > self.min_score:
            hands.append(OpenPoseHand(pose_score, list_of_vector4D(landmarks), Handedness.NONE))
        return hands

    def release(self):
        self.network = None

    def configure(self, args: Namespace):
        super().configure(args)
