from argparse import ArgumentParser, Namespace

import numpy as np
import vector

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.face.pose.HeadPoseEstimator import HeadPoseEstimator
from visiongraph.result.HeadPoseResult import HeadPoseResult


class AdasHeadPoseEstimator(HeadPoseEstimator):
    def __init__(self, device: str = "CPU"):
        model, weights = RepositoryAsset.openVino("head-pose-estimation-adas-0001")
        self.engine = VisionInferenceEngine(model, weights, device=device)

    def setup(self):
        self.engine.setup()

    def process(self, data: np.ndarray) -> HeadPoseResult:
        output = self.engine.process(data)
        return HeadPoseResult(vector.obj(
            x=float(output["angle_p_fc"][0][0]),
            y=float(output["angle_y_fc"][0][0]),
            z=float(output["angle_r_fc"][0][0])
        ))

    def _transform_result(self, result: HeadPoseResult, image: np.ndarray, roi: np.ndarray, xs: float, ys: float):
        pass

    def release(self):
        self.engine.release()

    def configure(self, args: Namespace):
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass
