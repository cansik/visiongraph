from argparse import ArgumentParser, Namespace

import numpy as np
from scipy.special import softmax

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.BaseClassifier import BaseClassifier
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.RoiEstimator import RoiEstimator
from visiongraph.result.HeadPoseResult import HeadPoseResult
from visiongraph.result.spatial.face.EyeOpenClosedResult import EyeOpenClosedResult


class EyeOpenClosedEstimator(RoiEstimator, BaseClassifier):
    def __init__(self, device: str = "AUTO"):
        super().__init__(0.5)
        model, weights = RepositoryAsset.openVino("open-closed-eye-0001-fp32")
        self.engine = OpenVinoEngine(model, weights, device=device)

        self.labels = ["closed", "open"]

    def setup(self):
        self.engine.setup()

    def process(self, data: np.ndarray) -> EyeOpenClosedResult:
        output = self.engine.process(data)

        probability = softmax(np.squeeze(output[self.engine.output_names[0]]))
        best_index = int(np.argmax(probability))

        return EyeOpenClosedResult(best_index, self.labels[best_index],
                                   float(probability[best_index]), probability)

    def _transform_result(self, result: HeadPoseResult, image: np.ndarray, roi: np.ndarray, xs: float, ys: float):
        pass

    def release(self):
        self.engine.release()

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
