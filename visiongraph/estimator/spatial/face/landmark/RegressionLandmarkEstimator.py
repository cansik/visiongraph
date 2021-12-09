from argparse import ArgumentParser, Namespace

import numpy as np

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.face.landmark.FaceLandmarkEstimator import FaceLandmarkEstimator
from visiongraph.result.spatial.face.RegressionFace import RegressionFace
from visiongraph.util.ResultUtils import list_of_vector4D


class RegressionLandmarkEstimator(FaceLandmarkEstimator):
    def __init__(self, min_score: float = 0.0, device: str = "CPU"):
        super().__init__(min_score)
        model, weights = RepositoryAsset.openVino("landmarks-regression-retail-0009")
        self.engine = VisionInferenceEngine(model, weights, 1, 3, 48, 48, device=device)

    def setup(self):
        self.engine.setup()

    def estimate(self, image: np.ndarray, **kwargs) -> RegressionFace:
        outputs = self.engine.estimate(image)
        output = outputs[self.engine.output_names[0]].reshape((-1, 2))
        result = []

        for point in output:
            result.append((float(point[0]), float(point[1]), 0.0, 1.0))

        return RegressionFace(1.0, list_of_vector4D(result))

    def _transform_result(self, result: RegressionFace, image: np.ndarray, roi: np.ndarray, xs: float, ys: float):
        hi, wi = image.shape[:2]
        hr, wr = roi.shape[:2]

        for i, lm in enumerate(result.landmarks):
            x = ((lm.x * wr) + xs) / float(wi)
            y = ((lm.y * hr) + ys) / float(hi)
            result.landmarks.x[i] = x
            result.landmarks.y[i] = y

    def release(self):
        self.engine.release()

    def configure(self, args: Namespace):
        self.engine.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
