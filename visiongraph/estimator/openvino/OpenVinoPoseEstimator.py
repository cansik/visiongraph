import logging
import math
import time
from abc import abstractmethod, ABC
from typing import Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.estimator.openvino.SyncInferencePipeline import SyncInferencePipeline
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.external.intel.models.model import Model
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.VectorUtils import list_of_vector4D


class OpenVinoPoseEstimator(PoseEstimator[COCOPose], ABC):
    """
    A class to estimate 2D pose from a single image using OpenVINO.

    :param model: The model for inference.
    :param weights: The weights for the model.
    :param target_size: The target size of the input image. Defaults to None.
    :param aspect_ratio: The default aspect ratio. Defaults to 16/9.
    :param min_score: The minimum score for pose estimation. Defaults to 0.5.
    :param auto_adjust_aspect_ratio: Whether to automatically adjust the aspect ratio. Defaults to True.
    :param device: The device for inference. Defaults to "AUTO".
    """

    def __init__(self, model: Asset, weights: Asset,
                 target_size: Optional[int] = None, aspect_ratio: float = 16 / 9, min_score: float = 0.5,
                 auto_adjust_aspect_ratio: bool = True, device: str = "AUTO"):
        super().__init__(min_score)
        self.model = model
        self.weights = weights
        self.aspect_ratio = aspect_ratio
        self.target_size = target_size

        self.auto_adjust_aspect_ratio = auto_adjust_aspect_ratio
        self.device = device

        self.pipeline: Optional[SyncInferencePipeline] = None
        self.ie_model: Optional[Model] = None

    def setup(self):
        """
        Prepare the model and weights, and create an IE model.
        """
        Asset.prepare_all(self.model, self.weights)

        self.ie_model = self._create_ie_model()
        self.pipeline = SyncInferencePipeline(self.ie_model, self.device)
        self.pipeline.setup()

    def process(self, data: np.ndarray) -> ResultList[COCOPose]:
        """
        Process the input image and estimate poses.

        :param data: The input image.

        :return: A list of estimated pose objects.
        """
        h, w = data.shape[:2]

        # auto-adjust aspect ratio
        ratio = w / h
        if self.auto_adjust_aspect_ratio and not math.isclose(ratio, self.aspect_ratio, rel_tol=0, abs_tol=0.001):
            self.adjust_aspect_ratio(ratio)

        # estimate on image
        key_points, scores = self.pipeline.process(data)

        poses = ResultList()
        for score, kps in zip(scores, key_points):
            # todo: maybe improve performance by not iterating but using np
            kp_score = np.average(kps[:, 2])

            if kp_score < self.min_score:
                continue

            landmarks = [(float(kp[0]) / w, float(kp[1]) / h, 0, float(kp[2])) for kp in kps]
            poses.append(COCOPose(kp_score, list_of_vector4D(landmarks)))

        return poses

    def release(self):
        """
        Release the pipeline and IE model.
        """
        self.pipeline.release()

    def adjust_aspect_ratio(self, ratio: float, timeout: float = 5.0):
        """
        Adjust the aspect ratio of the input image.

        :param ratio: The new aspect ratio.
        :param timeout: The timeout for adjusting the aspect ratio. Defaults to 5.0.
        """
        logging.warning(f"auto-adjusting aspect ratio to {ratio:.2f}")
        self.aspect_ratio = ratio

        # restart network
        self.release()
        self.setup()

        start = time.time()
        while not self.ie_model.is_ready() and time.time() - start < timeout:
            time.sleep(0.1)

        if not self.ie_model.is_ready():
            raise Exception("Adjusting aspect ratio did not work - Model is not ready!")

    @abstractmethod
    def _create_ie_model(self) -> Model:
        """
        Create the IE model for inference.

        :return: The created IE model.
        """
        pass
