from abc import abstractmethod, ABC
from typing import List, Optional

import numpy as np
from openvino.inference_engine import IECore

from visiongraph.data.Asset import Asset
from visiongraph.estimator.openvino.SyncInferencePipeline import SyncInferencePipeline
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.external.intel.model import Model
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.ResultUtils import list_of_vector4D


class OpenVinoPoseEstimator(PoseEstimator, ABC):
    def __init__(self, model: Asset, weights: Asset,
                 target_size: Optional[int] = None, aspect_ratio: float = 16/9, min_score: float = 0.5,
                 device: str = "CPU"):
        super().__init__(min_score)
        self.model = model
        self.weights = weights
        self.aspect_ratio = aspect_ratio
        self.target_size = target_size
        self.device = device

        self.ie = IECore()
        self.pipeline: Optional[SyncInferencePipeline] = None
        self.ie_model: Optional[Model] = None

    def setup(self):
        Asset.prepare_all(self.model, self.weights)

        self.ie_model = self._create_ie_model()
        self.pipeline = SyncInferencePipeline(self.ie_model, self.device, self.ie)
        self.pipeline.setup()

    def estimate(self, image: np.ndarray, **kwargs) -> List[COCOPose]:
        h, w = image.shape[:2]
        key_points, scores = self.pipeline.estimate(image)

        poses = []
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
