from argparse import ArgumentParser, Namespace

import numpy as np
import vector

from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.face.pose.HeadPoseEstimator import HeadPoseEstimator
from visiongraph.result.HeadPoseResult import HeadPoseResult


class AdasHeadPoseEstimator(HeadPoseEstimator):
    """
    A class to estimate head pose using the ADAS model.

    https://docs.openvino.ai/2024/omz_models_model_head_pose_estimation_adas_0001.html
    """

    def __init__(self, device: str = "AUTO"):
        """
        Initializes the AdasHeadPoseEstimator with the given device.

        :param device: The device to use for inference. Defaults to "AUTO".
        """
        model, weights = RepositoryAsset.openVino("head-pose-estimation-adas-0001")
        self.engine = OpenVinoEngine(model, weights, device=device)

    def setup(self):
        """
        Sets up the engine for inference.
        """
        self.engine.setup()

    def process(self, data: np.ndarray) -> HeadPoseResult:
        """
        Processes the input data and returns the head pose result.

        :param data: The input data to process.

        :return: The head pose result containing the estimated position.
        """
        output = self.engine.process(data)
        return HeadPoseResult(vector.obj(
            x=float(output["angle_p_fc"][0][0]),
            y=float(output["angle_y_fc"][0][0]),
            z=float(output["angle_r_fc"][0][0])
        ))

    def _transform_result(self, result: HeadPoseResult, image: np.ndarray, roi: np.ndarray, xs: float, ys: float):
        """
        Transforms the head pose result.

        :param result: The head pose result to transform.
        :param image: The input image.
        :param roi: The region of interest.
        :param xs: The x-coordinate.
        :param ys: The y-coordinate.
        """
        pass

    def release(self):
        """
        Releases the engine resources.
        """
        self.engine.release()

    def configure(self, args: Namespace):
        """
        Configures the estimator with the given arguments.

        :param args: The parser arguments.
        """
        pass

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds parameters to the parser.

        :param parser: The parser to add parameters to.
        """
        pass
