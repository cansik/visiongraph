from argparse import Namespace
from enum import Enum
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from visiongraph.estimator.spatial.InstanceSegmentationEstimator import InstanceSegmentationEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.InstanceSegmentationResult import InstanceSegmentationResult
from visiongraph.result.spatial.pose.BlazePose import BlazePose

_mp_selfie_segmentation = mp.solutions.selfie_segmentation


class SelfieSegmentationModel(Enum):
    """
    Enum to represent the type of selfie segmentation model.
    """

    General = 0
    Landscape = 1


class MediaPipeSelfieSegmentation(InstanceSegmentationEstimator[InstanceSegmentationResult]):
    """
    A class to utilize the MediaPipe selfie segmentation model for instance segmentation.
    """

    def __init__(self, model_type: SelfieSegmentationModel = SelfieSegmentationModel.General):
        """
        Initializes the MediaPipeSelfieSegmentation estimator.

        :param model_type: The type of selfie segmentation model. Defaults to SelfieSegmentationModel.General.
        """
        super().__init__(0.5)
        self.model_type = model_type

        self.network: Optional[_mp_selfie_segmentation.SelfieSegmentation] = None

    def setup(self):
        """
        Sets up the MediaPipe selfie segmentation network.

        Note:
            If the network is not created, it will be initialized with the specified model type.
        """
        if self.network is None:
            self.network = _mp_selfie_segmentation.SelfieSegmentation(model_selection=self.model_type.value)

    def process(self, data: np.ndarray) -> ResultList[BlazePose]:
        """
        Processes the input image using the MediaPipe selfie segmentation model.

        :param data: The input image as a numpy array.

        :return: A list of instance segmentation results.
        """
        # pre-process image
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        results = self.network.process(image)

        # use segmentation
        mask = results.segmentation_mask
        mask_uint8 = (mask * 255).astype(np.uint8)

        # todo: find components and combine them to the boundingbox
        box = BoundingBox2D(0, 0, 1, 1)
        return ResultList([InstanceSegmentationResult(0, "human", 1.0, mask_uint8, box)])

    def release(self):
        """
        Releases the MediaPipe selfie segmentation network.
        """
        self.network.close()

    def configure(self, args: Namespace):
        """
        Configures the estimator with the provided arguments.

        :param args: The configuration arguments.
        """
        super().configure(args)

    @staticmethod
    def create(model_type: SelfieSegmentationModel = SelfieSegmentationModel.General) -> "MediaPipeSelfieSegmentation":
        """
        Creates a new instance of the MediaPipe selfie segmentation estimator.

        :param model_type: The type of selfie segmentation model. Defaults to SelfieSegmentationModel.General.

        :return: A new instance of the MediaPipe selfie segmentation estimator.
        """
        return MediaPipeSelfieSegmentation(model_type)
