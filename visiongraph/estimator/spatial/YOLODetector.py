from enum import Enum
from typing import List

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.openvino.OpenVinoObjectDetector import OpenVinoObjectDetector
from visiongraph.external.intel.adapters.openvino_adapter import OpenvinoAdapter, create_core
from visiongraph.external.intel.models.detection_model import DetectionModel
from visiongraph.external.intel.models.yolo import YOLO, YoloV4, YOLOX, YOLOF


class YOLOArchitecture(Enum):
    YOLO = YOLO.__model__
    YOLOv4 = YoloV4.__model__
    YOLOF = YOLOF.__model__
    YOLOX = YOLOX.__model__


class YOLOConfig(Enum):
    YOLOv3_FP32 = (*RepositoryAsset.openVino("yolo-v3-tf-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLO)
    YOLOv3_FP16 = (*RepositoryAsset.openVino("yolo-v3-tf-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLO)
    YOLOv3_Tiny_FP32 = (*RepositoryAsset.openVino("yolo-v3-tiny-tf-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLO)
    YOLOv3_Tiny_FP16 = (*RepositoryAsset.openVino("yolo-v3-tiny-tf-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLO)

    YOLOv4_FP32 = (*RepositoryAsset.openVino("yolo-v4-tf-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLOv4)
    YOLOv4_FP16 = (*RepositoryAsset.openVino("yolo-v4-tf-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLOv4)
    YOLOv4_Tiny_FP32 = (*RepositoryAsset.openVino("yolo-v4-tiny-tf-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLOv4)
    YOLOv4_Tiny_FP16 = (*RepositoryAsset.openVino("yolo-v4-tiny-tf-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLOv4)

    YOLOF_FP32 = (*RepositoryAsset.openVino("yolof-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLOF)
    YOLOF_FP16 = (*RepositoryAsset.openVino("yolof-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLOF)

    YOLOX_Tiny_FP32 = (*RepositoryAsset.openVino("yolox-tiny-fp32"), COCO_80_LABELS, YOLOArchitecture.YOLOX)
    YOLOX_Tiny_FP16 = (*RepositoryAsset.openVino("yolox-tiny-fp16"), COCO_80_LABELS, YOLOArchitecture.YOLOX)


class YOLODetector(OpenVinoObjectDetector):
    def __init__(self, model: Asset, weights: Asset, labels: List[str], min_score: float = 0.5,
                 architecture: YOLOArchitecture = YOLOArchitecture.YOLOv4,
                 device: str = "CPU"):
        super().__init__(model, weights, labels, min_score, device)

        self.architecture = architecture

    def _create_ie_model(self) -> DetectionModel:
        model_adapter = OpenvinoAdapter(create_core(), self.model.path, device=self.device)

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

        return DetectionModel.create_model(self.architecture.value, model_adapter, config)

    @staticmethod
    def create(config: YOLOConfig = YOLOConfig.YOLOv4_Tiny_FP16) -> "YOLODetector":
        model, weights, labels, architecture = config.value
        return YOLODetector(model, weights, labels, architecture=architecture)
