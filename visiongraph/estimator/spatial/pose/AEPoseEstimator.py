from enum import Enum
from typing import Optional

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoPoseEstimator import OpenVinoPoseEstimator
from visiongraph.external.intel.hpe_associative_embedding import HpeAssociativeEmbedding
from visiongraph.external.intel.model import Model


class AEPoseConfig(Enum):
    EfficientHRNet_288_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0005-fp16"),)
    EfficientHRNet_288_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0005-fp32"),)
    EfficientHRNet_352_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0006-fp16"),)
    EfficientHRNet_352_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0006-fp32"),)
    EfficientHRNet_448_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0007-fp16"),)
    EfficientHRNet_448_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0007-fp32"),)


class AEPoseEstimator(OpenVinoPoseEstimator):
    def __init__(self, model: Asset, weights: Asset,
                 target_size: Optional[int] = None, aspect_ratio: float = 16 / 9, min_score: float = 0.1,
                 device: str = "CPU"):
        super().__init__(model, weights, target_size, aspect_ratio, min_score, device)

    def _create_ie_model(self) -> Model:
        return HpeAssociativeEmbedding(self.ie, self.model.path, target_size=self.target_size,
                                       aspect_ratio=self.aspect_ratio, prob_threshold=self.min_score)

    @staticmethod
    def create(config: AEPoseConfig = AEPoseConfig.EfficientHRNet_288_FP16) -> "AEPoseEstimator":
        model, weights = config.value
        return AEPoseEstimator(model, weights)
