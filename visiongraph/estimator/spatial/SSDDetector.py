from enum import Enum
from typing import List, Tuple

import openvino

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_90_LABELS
from visiongraph.estimator.openvino.OpenVinoObjectDetector import OpenVinoObjectDetector
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter
from visiongraph.external.intel.models.detection_model import DetectionModel
from visiongraph.external.intel.models.ssd import SSD

_PERSON_LABELS = ["person"]


def _person_net(name: str) -> Tuple:
    """
    Helper function to create a person detection model based on the input name.

    :param name: The name of the person detection model.

    :return: A tuple containing model and labels for the person detection.
    """
    return (*RepositoryAsset.openVino(name), _PERSON_LABELS)


class SSDConfig(Enum):
    """
    An enumeration of SSD configurations with pre-defined settings for specific models.
    """

    SSDLiteMobileNetV2_FP32 = (*RepositoryAsset.openVino("ssdlite_mobilenet_v2_fp32"), COCO_90_LABELS)

    PersonDetection_0200_256x256_FP16_INT8 = _person_net("person-detection-0200-fp16-int8")

    # Series of predefined person detection models for different resolutions and precisions
    PersonDetection_0200_256x256_FP16 = _person_net("person-detection-0200-fp16")
    PersonDetection_0200_256x256_FP32 = _person_net("person-detection-0200-fp32")

    PersonDetection_0201_384x384_FP16_INT8 = _person_net("person-detection-0201-fp16-int8")
    PersonDetection_0201_384x384_FP16 = _person_net("person-detection-0201-fp16")
    PersonDetection_0201_384x384_FP32 = _person_net("person-detection-0201-fp32")

    PersonDetection_0202_512x512_FP16_INT8 = _person_net("person-detection-0202-fp16-int8")
    PersonDetection_0202_512x512_FP16 = _person_net("person-detection-0202-fp16")
    PersonDetection_0202_512x512_FP32 = _person_net("person-detection-0202-fp32")

    PersonDetection_0203_864x480_FP16_INT8 = _person_net("person-detection-0203-fp16-int8")
    PersonDetection_0203_864x480_FP16 = _person_net("person-detection-0203-fp16")
    PersonDetection_0203_864x480_FP32 = _person_net("person-detection-0203-fp32")

    PersonDetection_Retail_0013_FP16_INT8 = _person_net("person-detection-retail-0013-fp16-int8")
    PersonDetection_Retail_0013_FP16 = _person_net("person-detection-retail-0013-fp16")
    PersonDetection_Retail_0013_FP32 = _person_net("person-detection-retail-0013-fp32")


class SSDDetector(OpenVinoObjectDetector):
    """
    A class representing an SSD detector based on OpenVino.
    """

    def __init__(self, model: Asset, weights: Asset, labels: List[str], min_score: float = 0.5, device: str = "AUTO"):
        """
        Initializes the SSDDetector with the model, weights, labels, minimum score, and device.

        :param model: The model asset to be used.
        :param weights: The weights asset for the model.
        :param labels: The list of labels for detection.
        :param min_score: The minimum score threshold for detection.
        :param device: The device to run the detector on (default is "AUTO").
        """
        super().__init__(model, weights, labels, min_score, device)

    def _create_ie_model(self) -> DetectionModel:
        """
        Creates the Inference Engine model for the detector.

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
        return SSD.create_model(SSD.__model__, adapter, config, preload=True)

    @staticmethod
    def create(config: SSDConfig = SSDConfig.SSDLiteMobileNetV2_FP32) -> "SSDDetector":
        """
        Creates an instance of SSDDetector based on the provided config.

        :param config: The configuration for the detector model (default is SSDLiteMobileNetV2_FP32).

        :return: An instance of the SSDDetector class.
        """
        model, weights, labels = config.value
        return SSDDetector(model, weights, labels)

    def _get_label(self, index: int):
        """
        Gets the label based on the index.

        :param index: The index of the label.

        :return: The label corresponding to the index.
        """
        return super()._get_label(index - 1)
