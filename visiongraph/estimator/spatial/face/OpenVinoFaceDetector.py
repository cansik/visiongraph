from enum import Enum
from typing import List, Dict, Tuple, Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.face.FaceDetector import FaceDetector
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.face.FaceDetectionResult import FaceDetectionResult


class OpenVinoFaceConfig(Enum):
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
    def __init__(self, model: Asset, weights: Asset, min_score: float = 0.5, device: str = "CPU"):
        super().__init__(min_score)

        self.width: Optional[int] = None
        self.height: Optional[int] = None

        self.engine = VisionInferenceEngine(model, weights, device=device)

    def setup(self):
        self.engine.setup()
        _, _, self.height, self.width = self.engine.first_input_shape

    def process(self, data: np.ndarray) -> ResultList[FaceDetectionResult]:
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
        self.engine.release()

    def _get_results(self, outputs: Dict[str, np.ndarray]) -> List[Tuple[float, float, float, float, float]]:
        results = []
        output = outputs[self.engine.output_names[1]]

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
        model, weights = config.value
        return OpenVinoFaceDetector(model, weights)
