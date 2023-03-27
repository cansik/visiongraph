import math
from enum import Enum
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.EfficientPose import EfficientPose
from visiongraph.util import VectorUtils, MathUtils

_BODY_PARTS = ['head_top', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax',
               'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle',
               'left_hip', 'left_knee', 'left_ankle']


class EfficientPoseEstimatorConfig(Enum):
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
    def __init__(self, model: Asset, weights: Asset,
                 min_score: float = 0.1, device: str = "AUTO"):
        super().__init__(min_score)

        self.engine = VisionInferenceEngine(model, weights,
                                            flip_channels=True, padding=True,
                                            device=device)

    def setup(self):
        self.engine.setup()

    def process(self, data: np.ndarray) -> ResultList[EfficientPose]:
        output_dict = self.engine.process(data)
        outputs = output_dict[self.engine.output_names[0]]
        padding_box: BoundingBox2D = output_dict.padding_box

        body_parts = self._extract_coordinates(outputs)

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
        self.engine.release()

    @staticmethod
    def _extract_coordinates(frame_output, blur=False):
        """
        Extract coordinates from supplied confidence maps.

        Args:
            frame_output: ndarray
                Numpy array of shape (h, w, c)
            blur: boolean
                Adds blur to the confidence map

        Returns:
            List of predicted coordinates for all c body parts in the frame the outputs are computed from.
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
        model, weights = config.value
        return EfficientPoseEstimator(model, weights)
