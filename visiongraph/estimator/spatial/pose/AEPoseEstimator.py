from enum import Enum
from typing import Optional

import openvino

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoPoseEstimator import OpenVinoPoseEstimator
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter
from visiongraph.external.intel.models.hpe_associative_embedding import HpeAssociativeEmbedding
from visiongraph.external.intel.models.model import Model


class AEPoseConfig(Enum):
    """
    Enumeration of available configurations for the pose estimation models.

    Each configuration corresponds to a specific pre-trained model and its weights.
    """
    EfficientHRNet_288_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0005-fp16"),)
    EfficientHRNet_288_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0005-fp32"),)
    EfficientHRNet_352_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0006-fp16"),)
    EfficientHRNet_352_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0006-fp32"),)
    EfficientHRNet_448_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0007-fp16"),)
    EfficientHRNet_448_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0007-fp32"),)


class AEPoseEstimator(OpenVinoPoseEstimator):
    """
    A pose estimator that utilizes OpenVINO for human pose estimation.

    Inherits from OpenVinoPoseEstimator and provides additional functionality
    specific to the AEPose model configurations.
    """

    def __init__(self, model: Asset, weights: Asset,
                 target_size: Optional[int] = None, aspect_ratio: float = 16 / 9, min_score: float = 0.1,
                 auto_adjust_aspect_ratio: bool = True, device: str = "AUTO"):
        """
        Initializes the AEPoseEstimator with specified parameters.

        :param model: The asset representing the pose estimation model.
        :param weights: The asset representing the weights of the model.
        :param target_size: The target size for the input image.
        :param aspect_ratio: The aspect ratio for the input image.
        :param min_score: The minimum confidence score for detected poses.
        :param auto_adjust_aspect_ratio: Whether to automatically adjust the aspect ratio.
        :param device: The device to use for inference (e.g., "AUTO", "CPU", "GPU").
        """
        super().__init__(model, weights, target_size, aspect_ratio, min_score, auto_adjust_aspect_ratio, device)

    def _create_ie_model(self) -> Model:
        """
        Creates the inference engine model for pose estimation.

        :return: The created model for inference.
        """
        config = {
            'target_size': self.target_size,
            'aspect_ratio': self.aspect_ratio,
            'confidence_threshold': self.min_score,
            'padding_mode': 'center',
            'delta': 0.5
        }

        core = openvino.Core()
        adapter = OpenvinoAdapter(core, self.model.path, device=self.device)
        return HpeAssociativeEmbedding.create_model(HpeAssociativeEmbedding.__model__, adapter, config, preload=True)

    @staticmethod
    def create(config: AEPoseConfig = AEPoseConfig.EfficientHRNet_288_FP16) -> "AEPoseEstimator":
        """
        Creates an instance of AEPoseEstimator using the specified configuration.

        :param config: The configuration to use for model and weights.

        :return: An instance of the AEPoseEstimator initialized with the specified configuration.
        """
        model, weights = config.value
        return AEPoseEstimator(model, weights)
