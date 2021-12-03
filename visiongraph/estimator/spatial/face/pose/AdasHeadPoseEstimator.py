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
        self.engine = VisionInferenceEngine(model, weights, 1, 3, 60, 60, device=device)

    def setup(self):
        self.engine.setup()

    def estimate(self, image: np.ndarray, **kwargs) -> HeadPoseResult:
        output = self.engine.estimate(image)
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
        self.engine.configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        pass