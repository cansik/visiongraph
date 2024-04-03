from enum import Enum
from typing import List, Optional

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine
from visiongraph.estimator.spatial.UltralyticsYOLODetector import UltralyticsYOLODetector


class YOLOv5Config(Enum):
    YOLOv5_N = RepositoryAsset("yolov5n.onnx"), COCO_80_LABELS
    YOLOv5_S = RepositoryAsset("yolov5s.onnx"), COCO_80_LABELS
    YOLOv5_M = RepositoryAsset("yolov5m.onnx"), COCO_80_LABELS
    YOLOv5_L = RepositoryAsset("yolov5l.onnx"), COCO_80_LABELS
    YOLOv5_X = RepositoryAsset("yolov5x.onnx"), COCO_80_LABELS


class YOLOv5Detector(UltralyticsYOLODetector):
    def __init__(self, *assets: Asset, labels: List[str], min_score: float = 0.3,
                 nms: bool = True, nms_threshold: float = 0.5,
                 nms_eta: Optional[float] = None, nms_top_k: Optional[int] = None,
                 engine: InferenceEngine = InferenceEngine.ONNX):
        super().__init__(*assets,
                         labels=labels,
                         min_score=min_score,
                         nms=nms,
                         nms_threshold=nms_threshold,
                         nms_eta=nms_eta,
                         nms_top_k=nms_top_k,
                         engine=engine)
        self.transpose_result = False

    @staticmethod
    def create(config: YOLOv5Config = YOLOv5Config.YOLOv5_S) -> "YOLOv5Detector":
        model, labels = config.value
        return YOLOv5Detector(model, labels=labels)
