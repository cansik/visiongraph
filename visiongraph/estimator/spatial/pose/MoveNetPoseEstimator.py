from enum import Enum
from typing import List, Tuple

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.ResultUtils import non_maximum_suppression
from visiongraph.util.VectorUtils import list_of_vector4D


class MoveNetConfig(Enum):
    """
    Enumeration for different configurations of the MoveNet model, including
    different architectures and precision levels.
    """
    MoveNet_Single_Lightning_FP16 = (*RepositoryAsset.openVino("movenet-single-lightning-fp16"), False)
    MoveNet_Single_Lightning_FP32 = (*RepositoryAsset.openVino("movenet-single-lightning-fp32"), False)

    MoveNet_Single_Thunder_FP16 = (*RepositoryAsset.openVino("movenet-single-thunder-fp16"), False)
    MoveNet_Single_Thunder_FP32 = (*RepositoryAsset.openVino("movenet-single-thunder-fp32"), False)

    MoveNet_MultiPose_192x192_FP32 = (*RepositoryAsset.openVino("movenet-multipose-192x192-fp32"), True)
    MoveNet_MultiPose_192x256_FP32 = (*RepositoryAsset.openVino("movenet-multipose-192x256-fp32"), True)
    MoveNet_MultiPose_256x256_FP32 = (*RepositoryAsset.openVino("movenet-multipose-256x256-fp32"), True)
    MoveNet_MultiPose_256x320_FP32 = (*RepositoryAsset.openVino("movenet-multipose-256x320-fp32"), True)
    MoveNet_MultiPose_320x320_FP32 = (*RepositoryAsset.openVino("movenet-multipose-320x320-fp32"), True)
    MoveNet_MultiPose_480x640_FP32 = (*RepositoryAsset.openVino("movenet-multipose-480x640-fp32"), True)
    MoveNet_MultiPose_736x1280_FP32 = (*RepositoryAsset.openVino("movenet-multipose-736x1280-fp32"), True)
    MoveNet_MultiPose_1280x1920_FP32 = (*RepositoryAsset.openVino("movenet-multipose-1280x1920-fp32"), True)


MOVE_NET_KEY_POINT_COUNT = 17
"""
Constant representing the number of key points used in the MoveNet model for pose estimation.
"""


class MoveNetPoseEstimator(PoseEstimator[COCOPose]):
    """
    MoveNetPoseEstimator is responsible for estimating poses using the MoveNet model.
    """

    def __init__(self, model: Asset, weights: Asset, multi_pose: bool = False,
                 min_score: float = 0.3, enable_nms: bool = False, iou_threshold: float = 0.4,
                 device: str = "AUTO"):
        """
        Initializes the MoveNetPoseEstimator with the given parameters.

        :param model: The model asset to be used for pose estimation.
        :param weights: The weights asset for the model.
        :param multi_pose: Flag to indicate if multi-pose estimation is enabled.
        :param min_score: Minimum score threshold for valid pose detection.
        :param enable_nms: Flag to enable Non-Maximum Suppression.
        :param iou_threshold: IOU threshold for NMS.
        :param device: The device to run the model on (e.g., "AUTO").
        """
        super().__init__(min_score)

        self.engine = OpenVinoEngine(model, weights, flip_channels=True, device=device)
        self.enable_nms = enable_nms
        self.iou_threshold = iou_threshold
        self.multi_pose = multi_pose

    def setup(self):
        """
        Sets up the engine for the pose estimator.
        """
        self.engine.setup()

    def process(self, data: np.ndarray) -> ResultList[COCOPose]:
        """
        Processes the input data and estimates poses.

        :param data: The input data for pose estimation.

        :return: A list of detected poses with their associated scores.
        """
        outputs = self.engine.process(data)
        output = outputs[self.engine.output_names[0]]

        key_points_with_scores = output[0]
        key_points_with_scores = np.squeeze(key_points_with_scores)

        if not self.multi_pose:
            key_points_with_scores = [key_points_with_scores]

        poses: ResultList[COCOPose] = ResultList()
        for key_points_with_score in key_points_with_scores:
            key_points: List[Tuple[float, float, float, float]] = []
            max_score = 0.0

            # keypoint
            for index in range(MOVE_NET_KEY_POINT_COUNT):
                if self.multi_pose:
                    x = float(key_points_with_score[(index * 3) + 1])
                    y = float(key_points_with_score[(index * 3) + 0])
                    score = float(key_points_with_score[(index * 3) + 2])
                else:
                    x = float(key_points_with_score[1][index])
                    y = float(key_points_with_score[0][index])
                    score = float(key_points_with_score[2][index])

                key_points.append((x, y, 0, score))

                if score > max_score:
                    max_score = score

            if max_score < self.min_score:
                continue

            poses.append(COCOPose(max_score, list_of_vector4D(key_points)))

        if self.enable_nms:
            poses = ResultList(non_maximum_suppression(poses, self.min_score, self.iou_threshold))

        return poses

    def release(self):
        """
        Releases resources held by the engine.
        """
        self.engine.release()

    @staticmethod
    def create(config: MoveNetConfig = MoveNetConfig.MoveNet_MultiPose_256x320_FP32) -> "MoveNetPoseEstimator":
        """
        Creates an instance of MoveNetPoseEstimator based on the provided configuration.

        :param config: The configuration for the MoveNet model.

        :return: An instance of the MoveNetPoseEstimator.
        """
        model, weights, multi_pose = config.value
        return MoveNetPoseEstimator(model, weights, multi_pose=multi_pose)
