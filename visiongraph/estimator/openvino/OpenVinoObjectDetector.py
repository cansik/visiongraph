from abc import abstractmethod, ABC
from typing import List, Optional

import numpy as np
from openvino.inference_engine import IECore

from visiongraph.data.Asset import Asset
from visiongraph.estimator.openvino.SyncInferencePipeline import SyncInferencePipeline
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.external.intel.model import Model
from visiongraph.external.intel.utils import Detection
from visiongraph.external.intel.yolo import YOLO, YoloV4
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class OpenVinoObjectDetector(ObjectDetector, ABC):
    def __init__(self, model: Asset, weights: Asset, labels: List[str], min_score: float, device: str = "CPU"):
        super().__init__(min_score)
        self.model = model
        self.weights = weights
        self.labels = labels
        self.device = device

        self.ie = IECore()
        self.pipeline: Optional[SyncInferencePipeline] = None
        self.ie_model: Optional[YOLO] = None

    def setup(self):
        Asset.prepare_all(self.model, self.weights)

        self.ie_model = self._create_ie_model()
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

    @abstractmethod
    def _create_ie_model(self) -> Model:
        pass
