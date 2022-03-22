from enum import Enum
from typing import List, Optional

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS, COCO_90_LABELS
from visiongraph.estimator.openvino.OpenVinoObjectDetector import OpenVinoObjectDetector
from visiongraph.external.intel.model import Model
from visiongraph.external.intel.ssd import SSD
from visiongraph.external.intel.utils import InputTransform

_PERSON_LABELS = ["person"]


def _person_net(name: str):
    return *RepositoryAsset.openVino(name), _PERSON_LABELS


class SSDConfig(Enum):
    SSDLiteMobileNetV2_FP32 = (*RepositoryAsset.openVino("ssdlite_mobilenet_v2_fp32"), COCO_90_LABELS)

    PersonDetection_0200_256x256_FP16_INT8 = _person_net("person-detection-0200-fp16-int8")
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
    def __init__(self, model: Asset, weights: Asset, labels: List[str],
                 keep_aspect_ratio: bool = False, min_score: float = 0.5,
                 reverse_input_channels: bool = False, mean_values: Optional[List] = None,
                 scale_values: Optional[List] = None, device: str = "CPU"):
        super().__init__(model, weights, labels, min_score, device)

        self.keep_aspect_ratio = keep_aspect_ratio
        self.reverse_input_channels = reverse_input_channels
        self.mean_values = mean_values
        self.scale_values = scale_values

    def _create_ie_model(self) -> Model:
        input_transform = InputTransform(self.reverse_input_channels, self.mean_values, self.scale_values)
        return SSD(self.ie, self.model.path, input_transform, self.labels, self.keep_aspect_ratio)

    @staticmethod
    def create(config: SSDConfig = SSDConfig.SSDLiteMobileNetV2_FP32) -> "SSDDetector":
        model, weights, labels = config.value
        return SSDDetector(model, weights, labels)

    def _get_label(self, index: int):
        return super()._get_label(index - 1)
