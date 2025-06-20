from enum import Enum
from typing import List

import openvino

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.openvino.OpenVinoObjectDetector import OpenVinoObjectDetector
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter
from visiongraph.external.intel.models.centernet import CenterNet
from visiongraph.external.intel.models.detection_model import DetectionModel


class CenterNetConfig(Enum):
    """
    Enum class for predefined CenterNet configurations.
    """
    CenterNet_FP16 = (*RepositoryAsset.openVino("ctdet_coco_dlav0_512-fp16"), COCO_80_LABELS)
    CenterNet_FP32 = (*RepositoryAsset.openVino("ctdet_coco_dlav0_512-fp32"), COCO_80_LABELS)


class CenterNetDetector(OpenVinoObjectDetector):
    """
    A class to represent a CenterNet Detector based on OpenVinoObjectDetector.
    """

    def __init__(self, model: Asset, weights: Asset, labels: List[str], min_score: float = 0.5, device: str = "AUTO"):
        """
        Initializes the CenterNetDetector with model, weights, labels, minimum score, and device.

        :param model: The model asset.
        :param weights: The weights asset.
        :param labels: List of label strings.
        :param min_score: Minimum score threshold (default is 0.5).
        :param device: Device to execute inference on (default is "AUTO").
        """
        super().__init__(model, weights, labels, min_score, device)

    def _create_ie_model(self) -> DetectionModel:
        """
        Creates the Inference Engine model for CenterNetDetector.

        :return: The created detection model.
        """
        config = {
            'resize_type': None,
            'mean_values': None,
            'scale_values': None,
            'reverse_input_channels': True,
            'path_to_labels': None,
            'confidence_threshold': self.min_score,
            'input_size': None,  # The CTPN specific
            'num_classes': None,  # The NanoDet and NanoDetPlus specific
        }

        core = openvino.Core()
        adapter = OpenvinoAdapter(core, self.model.path, device=self.device)
        return CenterNet.create_model(CenterNet.__model__, adapter, config, preload=True)

    @staticmethod
    def create(config: CenterNetConfig = CenterNetConfig.CenterNet_FP32) -> "CenterNetDetector":
        """
        Creates a CenterNetDetector based on the specified CenterNet configuration.

        :param config: CenterNet configuration (default is CenterNet_FP32).

        :return: The created CenterNetDetector object.
        """
        model, weights, labels = config.value
        return CenterNetDetector(model, weights, labels)
