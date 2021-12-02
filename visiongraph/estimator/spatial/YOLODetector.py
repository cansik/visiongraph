from enum import Enum
from typing import List, Optional

import numpy as np
from openvino.inference_engine import IECore

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.openvino.SyncInferencePipeline import SyncInferencePipeline
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.external.intel.utils import Detection
from visiongraph.external.intel.yolo import YOLO, YoloV4
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


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


class YOLODetector(ObjectDetector):
    def __init__(self, model: Asset, weights: Asset, labels: List[str],
                 min_score: float = 0.5, nms_threshold: float = 0.5,
                 architecture: YOLOArchitecture = YOLOArchitecture.YOLOv4, device: str = "CPU"):
        super().__init__(min_score)
        self.model = model
        self.weights = weights
        self.labels = labels
        self.device = device
        self.nms_threshold = nms_threshold
        self.architecture = architecture

        self.ie = IECore()
        self.pipeline: Optional[SyncInferencePipeline] = None
        self.ie_model: Optional[YOLO] = None

    def setup(self):
        model_class = YoloV4 if self.architecture == YOLOArchitecture.YOLOv4 else YOLO
        # download models
        Asset.prepare_all(self.model, self.weights)
        self.ie_model = model_class(self.ie, self.model.path, self.labels,
                                    threshold=self.min_score, iou_threshold=self.nms_threshold)
        self.pipeline = SyncInferencePipeline(self.ie_model, self.device, self.ie)
        self.pipeline.setup()

    def estimate(self, image: np.ndarray, **kwargs) -> List[ObjectDetectionResult]:
        h, w = image.shape[:2]
        output: List[Detection] = self.pipeline.estimate(image)
        return [ObjectDetectionResult(d.id, self.labels[d.id], d.score,
                                      BoundingBox2D(d.xmin / w, d.ymin / h,
                                                    (d.xmax - d.xmin) / w, (d.ymax - d.ymin) / h))
                for d in output]

    def release(self):
        self.pipeline.release()

    @staticmethod
    def create(config: YOLOConfig = YOLOConfig.YOLOv4_Tiny_FP16) -> "YOLODetector":
        model, weights, labels, architecture = config.value
        return YOLODetector(model, weights, labels, architecture=architecture)
