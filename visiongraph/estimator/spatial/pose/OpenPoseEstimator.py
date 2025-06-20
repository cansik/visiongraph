from enum import Enum
from typing import Optional

import openvino

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoPoseEstimator import OpenVinoPoseEstimator
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter
from visiongraph.external.intel.models.model import Model
from visiongraph.external.intel.models.open_pose import OpenPose


class OpenPoseConfig(Enum):
    """
    Enumeration for OpenPose model configurations with various precision options.
    """
    LightWeightOpenPose_INT8 = (*RepositoryAsset.openVino("human-pose-estimation-0001-int8"),)
    LightWeightOpenPose_FP16 = (*RepositoryAsset.openVino("human-pose-estimation-0001-fp16"),)
    LightWeightOpenPose_FP32 = (*RepositoryAsset.openVino("human-pose-estimation-0001-fp32"),)


class OpenPoseEstimator(OpenVinoPoseEstimator):
    """
    A class for estimating human poses using OpenPose with OpenVINO framework.
    """

    def __init__(self, model: Asset, weights: Asset,
                 target_size: Optional[int] = None, aspect_ratio: float = 16 / 9, min_score: float = 0.1,
                 auto_adjust_aspect_ratio: bool = True, device: str = "AUTO"):
        """
        Initializes the OpenPoseEstimator with the specified parameters.

        :param model: The model asset for pose estimation.
        :param weights: The weights asset for pose estimation.
        :param target_size: The target size for the input images.
        :param aspect_ratio: The aspect ratio for resizing.
        :param min_score: The minimum score threshold for detections.
        :param auto_adjust_aspect_ratio: Whether to automatically adjust the aspect ratio.
        :param device: The device to run the inference on (e.g., "AUTO", "CPU", "GPU").
        """
        super().__init__(model, weights, target_size, aspect_ratio, min_score, auto_adjust_aspect_ratio, device)

    def _create_ie_model(self) -> Model:
        """
        Creates the inference engine model for OpenPose.

        :return: The created OpenPose model.
        """
        config = {
            'target_size': self.target_size,
            'aspect_ratio': self.aspect_ratio,
            'confidence_threshold': self.min_score,
            'padding_mode': None,
            'delta': None
        }

        core = openvino.Core()
        adapter = OpenvinoAdapter(core, self.model.path, device=self.device)
        return OpenPose.create_model(OpenPose.__model__, adapter, config, preload=True)

    @staticmethod
    def create(config: OpenPoseConfig = OpenPoseConfig.LightWeightOpenPose_FP16) -> "OpenPoseEstimator":
        """
        Factory method to create an instance of OpenPoseEstimator using the specified configuration.

        :param config: The configuration to use for model and weights.

        :return: An instance of OpenPoseEstimator.
        """
        model, weights = config.value
        return OpenPoseEstimator(model, weights)
