from enum import Enum
from typing import Optional

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoPoseEstimator import OpenVinoPoseEstimator
from visiongraph.external.intel.model import Model
from visiongraph.external.intel.open_pose import OpenPose


class OpenPoseConfig(Enum):
    LightWeightOpenPose_INT8 = (*RepositoryAsset.openVino("human-pose-estimation-0001-int8"),)
    LightWeightOpenPose_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0001-fp16"),)
    LightWeightOpenPose_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0001-fp32"),)


class OpenPoseEstimator(OpenVinoPoseEstimator):
    def __init__(self, model: Asset, weights: Asset,
                 target_size: Optional[int] = None, aspect_ratio: float = 16 / 9, min_score: float = 0.1,
                 device: str = "CPU"):
        super().__init__(model, weights, target_size, aspect_ratio, min_score, device)

    def _create_ie_model(self) -> Model:
        return OpenPose(self.ie, self.model.path, target_size=self.target_size,
                        aspect_ratio=self.aspect_ratio, prob_threshold=self.min_score)

    @staticmethod
    def create(config: OpenPoseConfig = OpenPoseConfig.LightWeightOpenPose_FP16) -> "OpenPoseEstimator":
        model, weights = config.value
        return OpenPoseEstimator(model, weights)
