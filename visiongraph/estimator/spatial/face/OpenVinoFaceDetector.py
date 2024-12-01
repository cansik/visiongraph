from enum import Enum
from typing import List, Dict, Tuple, Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.face.FaceDetector import FaceDetector
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.FaceDetectionResult import FaceDetectionResult

BOXES_NAME = "boxes"


class OpenVinoFaceConfig(Enum):
    """
    Enumeration of available OpenVino face detection model configurations.
    """
    MobileNetV2_256_FP16_INT8 = RepositoryAsset.openVino("face-detection-0200-fp16-int8")
    MobileNetV2_256_FP16 = RepositoryAsset.openVino("face-detection-0200-fp16")
    MobileNetV2_256_FP32 = RepositoryAsset.openVino("face-detection-0200-fp32")
    MobileNetV2_384_FP16_INT8 = RepositoryAsset.openVino("face-detection-0202-fp16-int8")
    MobileNetV2_384_FP16 = RepositoryAsset.openVino("face-detection-0202-fp16")
    MobileNetV2_384_FP32 = RepositoryAsset.openVino("face-detection-0202-fp32")
    MobileNetV2_448_FP16_INT8 = RepositoryAsset.openVino("face-detection-0204-fp16-int8")
    MobileNetV2_448_FP16 = RepositoryAsset.openVino("face-detection-0204-fp16")
    MobileNetV2_448_FP32 = RepositoryAsset.openVino("face-detection-0204-fp32")
    MobileNetV2_416_FP16_INT8 = RepositoryAsset.openVino("face-detection-0205-fp16-int8")
    MobileNetV2_416_FP16 = RepositoryAsset.openVino("face-detection-0205-fp16")
    MobileNetV2_416_FP32 = RepositoryAsset.openVino("face-detection-0205-fp32")
    MobileNetV2_640_FP16_INT8 = RepositoryAsset.openVino("face-detection-0206-fp16-int8")
    MobileNetV2_640_FP16 = RepositoryAsset.openVino("face-detection-0206-fp16")
    MobileNetV2_640_FP32 = RepositoryAsset.openVino("face-detection-0206-fp32")


class OpenVinoFaceDetector(FaceDetector[FaceDetectionResult]):
    """
    OpenVino implementation of a face detector using pre-trained models.

    Inherits from the FaceDetector class and uses OpenVino for inference.
    """

    def __init__(self, model: Asset, weights: Asset, min_score: float = 0.5, device: str = "AUTO"):
        """
        Initializes the OpenVinoFaceDetector with the specified model and weights.

        :param model: The model asset to be used for face detection.
        :param weights: The weights asset associated with the model.
        :param min_score: The minimum score threshold for detections. Defaults to 0.5.
        :param device: The device to be used for inference. Defaults to "AUTO".
        """
        super().__init__(min_score)

        self.width: Optional[int] = None
        self.height: Optional[int] = None

        self.engine = OpenVinoEngine(model, weights, device=device)

    def setup(self):
        """
        Sets up the OpenVino engine and retrieves the input dimensions.
        """
        self.engine.setup()
        _, _, self.height, self.width = self.engine.first_input_shape

    def process(self, data: np.ndarray) -> ResultList[FaceDetectionResult]:
        """
        Processes the input data to detect faces and returns the results.

        :param data: The input image data in which faces need to be detected.

        :return: A list of detected face results.
        """
        output = self._get_results(self.engine.process(data))

        results = ResultList()
        for score, xmin, ymin, xmax, ymax in output:
            if score < self.min_score:
                continue

            w = xmax - xmin
            h = ymax - ymin

            detection = FaceDetectionResult(score, BoundingBox2D(xmin, ymin, w, h))
            results.append(detection)

        return results

    def release(self):
        """
        Releases resources held by the OpenVino engine.
        """
        self.engine.release()

    def _get_results(self, outputs: Dict[str, np.ndarray]) -> List[Tuple[float, float, float, float, float]]:
        results = []

        output_name = BOXES_NAME if BOXES_NAME in self.engine.output_names else self.engine.output_names[0]
        output: np.ndarray = outputs[output_name]
        output = output.squeeze()

        for obj in output:
            score = float(obj[4])
            if score > self.min_score:
                xmin = float(obj[0]) / self.width
                ymin = float(obj[1]) / self.height
                xmax = float(obj[2]) / self.width
                ymax = float(obj[3]) / self.height

                results.append((score, xmin, ymin, xmax, ymax))

        return results

    @staticmethod
    def create(config: OpenVinoFaceConfig = OpenVinoFaceConfig.MobileNetV2_416_FP32) -> "OpenVinoFaceDetector":
        """
        Creates an instance of OpenVinoFaceDetector based on the specified configuration.

        :param config: The configuration to use for the face detector. Defaults to MobileNetV2_416_FP32.

        :return: An instance of OpenVinoFaceDetector.
        """
        model, weights = config.value
        return OpenVinoFaceDetector(model, weights)
