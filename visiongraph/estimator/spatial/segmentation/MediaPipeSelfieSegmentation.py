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
    General = 0
    Landscape = 1


class MediaPipeSelfieSegmentation(InstanceSegmentationEstimator[InstanceSegmentationResult]):
    def __init__(self, model_type: SelfieSegmentationModel = SelfieSegmentationModel.General):
        super().__init__(0.5)
        self.model_type = model_type

        self.network: Optional[_mp_selfie_segmentation.SelfieSegmentation] = None

    def setup(self):
        if self.network is None:
            self.network = _mp_selfie_segmentation.SelfieSegmentation(model_selection=self.model_type.value)

    def process(self, data: np.ndarray) -> ResultList[BlazePose]:
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
        self.network.close()

    def configure(self, args: Namespace):
        super().configure(args)

    @staticmethod
    def create(model_type: SelfieSegmentationModel = SelfieSegmentationModel.General) -> "MediaPipeSelfieSegmentation":
        return MediaPipeSelfieSegmentation(model_type)
