from enum import Enum
from typing import List

import openvino

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.openvino.OpenVinoObjectDetector import OpenVinoObjectDetector
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter
from visiongraph.external.intel.models.detection_model import DetectionModel
from visiongraph.external.intel.models.detr import DETR


class DETRConfig(Enum):
    """
    An enumeration class representing different configurations for the DETR model.
    """
    DETR_Resnet50_FP16 = (*RepositoryAsset.openVino("detr-resnet50-fp16"), COCO_80_LABELS)
    DETR_Resnet50_FP32 = (*RepositoryAsset.openVino("detr-resnet50-fp32"), COCO_80_LABELS)


class DETRDetector(OpenVinoObjectDetector):
    """
    A class representing a Detector based on the DETR model.
    Inherits from OpenVinoObjectDetector.
    """

    def __init__(self, model: Asset, weights: Asset, labels: List[str], min_score: float = 0.5, device: str = "AUTO"):
        """
        Initializes the DETRDetector object with the provided model, weights, labels, minimum score, and device.

        :param model: The model Asset for the detector.
        :param weights: The weights Asset for the detector.
        :param labels: The list of labels for detection.
        :param min_score: The minimum score threshold for detection (default is 0.5).
        :param device: The device to run inference on (default is "AUTO").
        """

        super().__init__(model, weights, labels, min_score, device)

    def _create_ie_model(self) -> DetectionModel:
        """
        Create an Inference Engine model for DETR detector.

        :return: The DetectionModel for DETR object detection.
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
        return DETR.create_model(DETR.__model__, adapter, config, preload=True)

    @staticmethod
    def create(config: DETRConfig = DETRConfig.DETR_Resnet50_FP32) -> "DETRDetector":
        """
        Static method to create a DETRDetector object based on the given configuration.

        :param config: The configuration for the DETR detector (default is DETR_Resnet50_FP32).

        :return: A new instance of DETRDetector based on the provided configuration.
        """
        model, weights, labels = config.value
        return DETRDetector(model, weights, labels)

    def _get_label(self, index: int):
        """
        Get the label for a given index.

        :param index: The index of the label.

        :return: The label corresponding to the index.
        """
        return super()._get_label(index - 1)
