from abc import abstractmethod, ABC
from typing import List, Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.estimator.openvino.SyncInferencePipeline import SyncInferencePipeline
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.external.intel.models.detection_model import DetectionModel
from visiongraph.external.intel.models.model import Model
from visiongraph.external.intel.models.utils import Detection
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult


class OpenVinoObjectDetector(ObjectDetector[ObjectDetectionResult], ABC):
    """
    A class that uses the Intel OpenVINO framework to create an object detector.
    It provides a pre-built pipeline for inference and automatic model preparation.
    """

    def __init__(self,
                 model: Asset,
                 weights: Asset,
                 labels: List[str],
                 min_score: float,
                 device: str = "AUTO"):
        """
        Initializes the OpenVinoObjectDetector object.

        :param model: The model to be used for detection.
        :param weights: The weights for the model.
        :param labels: A list of labels corresponding to each class in the model.
        :param min_score: The minimum score required for a detection to be considered valid.
        :param device: The device on which the pipeline will run. Defaults to "AUTO".
        """
        super().__init__(min_score)
        self.model = model
        self.weights = weights
        self.labels = labels
        self.device = device

        self.pipeline: Optional[SyncInferencePipeline] = None
        self.ie_model: Optional[Model] = None

    def setup(self):
        """
        Sets up the OpenVino pipeline by preparing all assets and creating an IE model.
        """
        Asset.prepare_all(self.model, self.weights)

        self.ie_model = self._create_ie_model()
        self.ie_model.labels = self.labels

        self.pipeline = SyncInferencePipeline(self.ie_model, self.device)
        self.pipeline.setup()

    def process(self, data: np.ndarray) -> ResultList[ObjectDetectionResult]:
        """
        Processes the input data using the OpenVino pipeline.

        :param data: The input image or video frame.

        :return: A list of object detection results.
        """
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
        """
        Releases the OpenVino pipeline.
        """
        self.pipeline.release()

    @abstractmethod
    def _create_ie_model(self) -> DetectionModel:
        """
        Creates an IE model from the provided detection model.

        :return: The created IE model.
        """
        pass

    def _get_label(self, index: int):
        """
        Retrieves the label corresponding to the given class index.

        :param index: The class index.

        :return: The label for the given class index. If the index is out of range, returns "NoLabelFound".
        """
        if index < len(self.labels):
            return self.labels[index]
        else:
            return "NoLabelFound"
