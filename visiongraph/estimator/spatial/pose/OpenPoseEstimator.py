from enum import Enum
from typing import Optional

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoPoseEstimator import OpenVinoPoseEstimator
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter, create_core
from visiongraph.external.intel.models.model import Model
from visiongraph.external.intel.models.open_pose import OpenPose


class OpenPoseConfig(Enum):
    LightWeightOpenPose_INT8 = (*RepositoryAsset.openVino("human-pose-estimation-0001-int8"),)
    LightWeightOpenPose_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0001-fp16"),)
    LightWeightOpenPose_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0001-fp32"),)


class OpenPoseEstimator(OpenVinoPoseEstimator):
    def __init__(self, model: Asset, weights: Asset,
                 target_size: Optional[int] = None, aspect_ratio: float = 16 / 9, min_score: float = 0.1,
                 auto_adjust_aspect_ratio: bool = True, device: str = "CPU"):
        super().__init__(model, weights, target_size, aspect_ratio, min_score, auto_adjust_aspect_ratio, device)

    def _create_ie_model(self) -> Model:
        model_adapter = OpenvinoAdapter(create_core(), self.model.path, device=self.device)

        config = {
            'target_size': self.target_size,
            'aspect_ratio': self.aspect_ratio,
            'confidence_threshold': self.min_score,
            'padding_mode': None,
            'delta': None
        }

        return OpenPose.create_model("openpose", model_adapter, config)

    @staticmethod
    def create(config: OpenPoseConfig = OpenPoseConfig.LightWeightOpenPose_FP16) -> "OpenPoseEstimator":
        model, weights = config.value
        return OpenPoseEstimator(model, weights)
