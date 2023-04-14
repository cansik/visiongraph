from enum import Enum
from typing import List, Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.util.ResultUtils import non_maximum_suppression


class YOLOv5Config(Enum):
    YOLOv5_N = RepositoryAsset("yolov5n.onnx"), COCO_80_LABELS
    YOLOv5_S = RepositoryAsset("yolov5s.onnx"), COCO_80_LABELS
    YOLOv5_M = RepositoryAsset("yolov5m.onnx"), COCO_80_LABELS
    YOLOv5_L = RepositoryAsset("yolov5l.onnx"), COCO_80_LABELS
    YOLOv5_X = RepositoryAsset("yolov5x.onnx"), COCO_80_LABELS


class YOLOv5Detector(ObjectDetector):
    def __init__(self, *assets: Asset, labels: List[str], min_score: float = 0.3,
                 nms: bool = True, nms_threshold: float = 0.5,
                 nms_eta: Optional[float] = None, nms_top_k: Optional[int] = None,
                 engine: InferenceEngine = InferenceEngine.ONNX):
        super().__init__(min_score)
        self.engine = InferenceEngineFactory.create(engine, assets,
                                                    flip_channels=True,
                                                    scale=255.0,
                                                    padding=True)
        # set padding color
        self.engine.padding_color = (125, 125, 125)

        self.labels: List[str] = labels
        self.nms_threshold: float = nms_threshold
        self.nms: bool = nms
        self.nms_eta = nms_eta
        self.nms_top_k = nms_top_k

    def setup(self):
        self.engine.setup()

    def process(self, image: np.ndarray) -> ResultList[ObjectDetectionResult]:
        output = self.engine.process(image)
        detections = output[self.engine.output_names[0]]
        detections = np.squeeze(detections)

        # filter detection min score
        detections = detections[np.where(detections[:, 4] > self.min_score)]

        h, w = self.engine.first_input_shape[2:]

        # create result list
        results = ResultList()
        for pred in detections:
            det_bbox, score, det_label = pred[0:4], pred[4], pred[5:]

            # find label
            label_index = int(np.argmax(det_label))

            # process bounding box
            wh = det_bbox[2:]
            xy = det_bbox[:2]
            xy -= wh * 0.5
            bbox = BoundingBox2D(xy[0], xy[1], wh[0], wh[1]).scale(1 / w, 1 / h)

            detection = ObjectDetectionResult(label_index, self.labels[label_index], score, bbox)
            detection.map_coordinates(output.image_size, Size2D.from_image(image), src_roi=output.padding_box)
            results.append(detection)

        if self.nms:
            results = ResultList(non_maximum_suppression(results, self.min_score, self.nms_threshold,
                                                         self.nms_eta, self.nms_top_k))
        return results

    def release(self):
        self.engine.release()

    @staticmethod
    def create(config: YOLOv5Config = YOLOv5Config.YOLOv5_S) -> "YOLOv5Detector":
        model, labels = config.value
        return YOLOv5Detector(model, labels=labels)
