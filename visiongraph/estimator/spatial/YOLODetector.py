from enum import Enum
from typing import List

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.openvino.OpenVinoObjectDetector import OpenVinoObjectDetector
from visiongraph.external.intel.model import Model
from visiongraph.external.intel.yolo import YOLO, YoloV4


class YOLOArchitecture(Enum):
    YOLO = 0x1
    YOLOv4 = 0x2


class YOLOConfig(Enum):
    YOLOv3_FP32 = (*RepositoryAsset.openVino("yolo-v3-tf-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLO)
    YOLOv3_FP16 = (*RepositoryAsset.openVino("yolo-v3-tf-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLO)
    YOLOv3_Tiny_FP32 = (*RepositoryAsset.openVino("yolo-v3-tiny-tf-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLO)
    YOLOv3_Tiny_FP16 = (*RepositoryAsset.openVino("yolo-v3-tiny-tf-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLO)
    YOLOv4_FP32 = (*RepositoryAsset.openVino("yolo-v4-tf-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLOv4)
    YOLOv4_FP16 = (*RepositoryAsset.openVino("yolo-v4-tf-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLOv4)
    YOLOv4_Tiny_FP32 = (*RepositoryAsset.openVino("yolo-v4-tiny-tf-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLOv4)
    YOLOv4_Tiny_FP16 = (*RepositoryAsset.openVino("yolo-v4-tiny-tf-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLOv4)


class YOLODetector(OpenVinoObjectDetector):
    def __init__(self, model: Asset, weights: Asset, labels: List[str], min_score: float = 0.5,
                 nms_threshold: float = 0.5, architecture: YOLOArchitecture = YOLOArchitecture.YOLOv4,
                 device: str = "CPU"):
        super().__init__(model, weights, labels, min_score, device)
        self.nms_threshold = nms_threshold
        self.architecture = architecture

    def _create_ie_model(self) -> Model:
        model_class = YoloV4 if self.architecture == YOLOArchitecture.YOLOv4 else YOLO
        return model_class(self.ie, self.model.path, self.labels,
                           threshold=self.min_score, iou_threshold=self.nms_threshold)

    @staticmethod
    def create(config: YOLOConfig = YOLOConfig.YOLOv4_Tiny_FP16) -> "YOLODetector":
        model, weights, labels, architecture = config.value
        return YOLODetector(model, weights, labels, architecture=architecture)
