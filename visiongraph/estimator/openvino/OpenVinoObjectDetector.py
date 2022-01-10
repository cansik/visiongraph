from abc import abstractmethod, ABC
from typing import List, Optional

import numpy as np
from openvino.inference_engine import IECore

from visiongraph.data.Asset import Asset
from visiongraph.estimator.openvino.SyncInferencePipeline import SyncInferencePipeline
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.external.intel.model import Model
from visiongraph.external.intel.utils import Detection
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class OpenVinoObjectDetector(ObjectDetector[ObjectDetectionResult], ABC):
    def __init__(self, model: Asset, weights: Asset, labels: List[str], min_score: float, device: str = "CPU"):
        super().__init__(min_score)
        self.model = model
        self.weights = weights
        self.labels = labels
        self.device = device

        self.ie: Optional[IECore] = None
        self.pipeline: Optional[SyncInferencePipeline] = None
        self.ie_model: Optional[Model] = None

    def setup(self):
        Asset.prepare_all(self.model, self.weights)

        if self.ie is None:
            self.ie = IECore()

        self.ie_model = self._create_ie_model()
        self.pipeline = SyncInferencePipeline(self.ie_model, self.device, self.ie)
        self.pipeline.setup()

    def process(self, data: np.ndarray) -> ResultList[ObjectDetectionResult]:
        h, w = data.shape[:2]
        output: List[Detection] = self.pipeline.process(data)

        return ResultList([ObjectDetectionResult(int(d.id),
                                                 self._get_label(int(d.id)),
                                                 float(d.score),
                                                 BoundingBox2D(float(d.xmin) / w,
                                                               float(d.ymin) / h,
                                                               float(d.xmax - d.xmin) / w,
                                                               float(d.ymax - d.ymin) / h))
                           for d in output if float(d.score) >= self.min_score])

    def release(self):
        self.pipeline.release()

    @abstractmethod
    def _create_ie_model(self) -> Model:
        pass

    def _get_label(self, index: int):
        if index < len(self.labels):
            return self.labels[index]
        else:
            return "NoLabelFound"
