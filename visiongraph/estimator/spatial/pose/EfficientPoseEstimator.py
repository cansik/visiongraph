import math
from enum import Enum
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.types.InputShapeOrder import InputShapeOrder
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.EfficientPose import EfficientPose
from visiongraph.util import VectorUtils, MathUtils

_BODY_PARTS = ['head_top', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax',
               'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle',
               'left_hip', 'left_knee', 'left_ankle']


class EfficientPoseEstimatorConfig(Enum):
    """
    Enumeration for EfficientPose estimator configurations with associated model and weights.
    """
    EFFICIENT_POSE_I_FP16 = RepositoryAsset.openVino("EfficientPoseI-fp16")
    EFFICIENT_POSE_I_FP32 = RepositoryAsset.openVino("EfficientPoseI-fp32")
    EFFICIENT_POSE_II_FP16 = RepositoryAsset.openVino("EfficientPoseII-fp16")
    EFFICIENT_POSE_II_FP32 = RepositoryAsset.openVino("EfficientPoseII-fp32")
    EFFICIENT_POSE_III_FP16 = RepositoryAsset.openVino("EfficientPoseIII-fp16")
    EFFICIENT_POSE_III_FP32 = RepositoryAsset.openVino("EfficientPoseIII-fp32")
    EFFICIENT_POSE_IV_FP16 = RepositoryAsset.openVino("EfficientPoseIV-fp16")
    EFFICIENT_POSE_IV_FP32 = RepositoryAsset.openVino("EfficientPoseIV-fp32")
    EFFICIENT_POSE_RT_FP16 = RepositoryAsset.openVino("EfficientPoseRT-fp16")
    EFFICIENT_POSE_RT_FP32 = RepositoryAsset.openVino("EfficientPoseRT-fp32")

    EFFICIENT_POSE_II_LITE_FP16 = RepositoryAsset.openVino("EfficientPoseII_LITE-fp16")
    EFFICIENT_POSE_II_LITE_FP32 = RepositoryAsset.openVino("EfficientPoseII_LITE-fp32")
    EFFICIENT_POSE_I_LITE_FP16 = RepositoryAsset.openVino("EfficientPoseI_LITE-fp16")
    EFFICIENT_POSE_I_LITE_FP32 = RepositoryAsset.openVino("EfficientPoseI_LITE-fp32")
    EFFICIENT_POSE_RT_LITE_FP16 = RepositoryAsset.openVino("EfficientPoseRT_LITE-fp16")
    EFFICIENT_POSE_RT_LITE_FP32 = RepositoryAsset.openVino("EfficientPoseRT_LITE-fp32")


class EfficientPoseEstimator(PoseEstimator[EfficientPose]):
    """
    A pose estimator that utilizes the EfficientPose model for estimating human poses in images.

    :param model: The model asset to be used for inference.
    :param weights: The weights asset to be used for inference.
    :param min_score: Minimum score threshold for detections. Defaults to 0.1.
    :param device: The device to run the inference on. Defaults to "AUTO".
    """

    def __init__(self, model: Asset, weights: Asset,
                 min_score: float = 0.1, device: str = "AUTO"):
        super().__init__(min_score)

        self.engine = OpenVinoEngine(model, weights,
                                     flip_channels=True, padding=True,
                                     device=device)
        self.engine.order = InputShapeOrder.NWHC

    def setup(self):
        """
Sets up the inference engine.
"""
        self.engine.setup()

    def process(self, data: np.ndarray) -> ResultList[EfficientPose]:
        """
        Processes the input image data to extract pose information.

        :param data: The input image data in the form of a numpy array.

        :return: A list containing detected poses and their scores.
        """
        output_dict = self.engine.process(data)
        outputs = output_dict[self.engine.output_names[0]]
        padding_box: BoundingBox2D = output_dict.padding_box

        # transpose data to nchw from nhwc
        outputs_nchw = np.transpose(outputs, (0, 3, 1, 2))
        body_parts = self._extract_coordinates(outputs_nchw)

        landmarks: List[Tuple[float, float, float, float]] = []
        max_score = 0.0
        for name, x, y, score in body_parts:
            landmarks.append((
                MathUtils.map_value(x - padding_box.x_min, 0, padding_box.width, 0, 1),
                MathUtils.map_value(y - padding_box.y_min, 0, padding_box.height, 0, 1),
                0.0, float(score)))

            if max_score < score:
                max_score = float(score)

        return ResultList([EfficientPose(max_score, VectorUtils.list_of_vector4D(landmarks))])

    def release(self):
        """
Releases resources held by the inference engine.
"""
        self.engine.release()

    @staticmethod
    def _extract_coordinates(frame_output, blur=False):
        """
        Extract coordinates from supplied confidence maps.

        :param frame_output: ndarray
        :param blur: boolean

        """
        # Fetch output resolution
        output_height, output_width = frame_output.shape[2:]

        # Initialize coordinates
        frame_coords = []

        # Iterate over body parts
        for i in range(frame_output.shape[1]):

            # Find peak point
            conf = frame_output[0, i, ...]
            if blur:
                conf = gaussian_filter(conf, sigma=1.)

            max_index = np.argmax(conf)
            peak_y = float(math.floor(max_index / output_width))
            peak_x = max_index % output_width
            confidence = conf[int(peak_y), int(peak_x)]

            # Normalize coordinates
            peak_x /= output_width
            peak_y /= output_height

            frame_coords.append((_BODY_PARTS[i], peak_x, peak_y, confidence))

        return frame_coords

    @staticmethod
    def create(config: EfficientPoseEstimatorConfig
               = EfficientPoseEstimatorConfig.EFFICIENT_POSE_I_FP32) -> "EfficientPoseEstimator":
        """
        Creates an instance of EfficientPoseEstimator with specified configuration.

        :param config: Configuration for the estimator.

        :return: An instance of the pose estimator.
        """
        model, weights = config.value
        return EfficientPoseEstimator(model, weights)
