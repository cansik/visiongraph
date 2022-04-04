import logging
import math
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
    def __init__(self, model: Asset, weights: Asset,
                 target_size: Optional[int] = None, aspect_ratio: float = 16 / 9, min_score: float = 0.5,
                 auto_adjust_aspect_ratio: bool = True, device: str = "CPU"):
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
        Asset.prepare_all(self.model, self.weights)

        self.ie_model = self._create_ie_model()
        self.pipeline = SyncInferencePipeline(self.ie_model, self.device)
        self.pipeline.setup()

    def process(self, data: np.ndarray) -> ResultList[COCOPose]:
        h, w = data.shape[:2]

        # auto-adjust aspect ratio
        ratio = w / h
        if not math.isclose(ratio, self.aspect_ratio, rel_tol=0, abs_tol=0.001) and self.auto_adjust_aspect_ratio:
            logging.warning(f"auto-adjusting aspect ratio to {ratio:.2f}")
            self.aspect_ratio = ratio

            # restart network
            self.release()
            self.setup()

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
        self.pipeline.release()

    @abstractmethod
    def _create_ie_model(self) -> Model:
        pass
