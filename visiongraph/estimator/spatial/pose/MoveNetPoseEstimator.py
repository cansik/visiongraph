from enum import Enum
from typing import List, Tuple

import numpy as np
import vector

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.spatial.pose.MoveNetPose import MoveNetPose
from visiongraph.result.spatial.pose.PoseLandmarkResult import PoseLandmarkResult
from visiongraph.util.ResultUtils import list_of_vector4D


class MoveNetConfig(Enum):
    MoveNet_MultiPose_192x192_FP32 = (*RepositoryAsset.openVino("movenet-multipose-192x192-fp32"), 192, 192)
    MoveNet_MultiPose_192x256_FP32 = (*RepositoryAsset.openVino("movenet-multipose-192x256-fp32"), 192, 256)
    MoveNet_MultiPose_256x256_FP32 = (*RepositoryAsset.openVino("movenet-multipose-256x256-fp32"), 256, 256)
    MoveNet_MultiPose_256x320_FP32 = (*RepositoryAsset.openVino("movenet-multipose-256x320-fp32"), 256, 320)
    MoveNet_MultiPose_320x320_FP32 = (*RepositoryAsset.openVino("movenet-multipose-320x320-fp32"), 320, 320)
    MoveNet_MultiPose_480x640_FP32 = (*RepositoryAsset.openVino("movenet-multipose-480x640-fp32"), 480, 640)
    MoveNet_MultiPose_736x1280_FP32 = (*RepositoryAsset.openVino("movenet-multipose-736x1280-fp32"), 736, 1280)
    MoveNet_MultiPose_1280x1920_FP32 = (*RepositoryAsset.openVino("movenet-multipose-1280x1920-fp32"), 1280, 1920)


MOVE_NET_KEY_POINT_COUNT = 17


class MoveNetPoseEstimator(PoseEstimator):
    def __init__(self, model: Asset, weights: Asset, width: int, height: int,
                 min_score: float = 0.3, device: str = "CPU"):
        super().__init__(min_score)

        self.engine = VisionInferenceEngine(model, weights, 1, 3, width, height, device=device)

    def setup(self):
        self.engine.setup()

    def estimate(self, image: np.ndarray, **kwargs) -> List[PoseLandmarkResult]:
        outputs = self.engine.estimate(image)
        output = outputs[self.engine.output_names[0]]

        key_points_with_scores = output[0]
        key_points_with_scores = np.squeeze(key_points_with_scores)

        poses: List[MoveNetPose] = []
        for key_points_with_score in key_points_with_scores:
            key_points: List[Tuple[float, float, float, float]] = []
            max_score = 0.0

            # keypoint
            for index in range(MOVE_NET_KEY_POINT_COUNT):
                x = float(key_points_with_score[(index * 3) + 1])
                y = float(key_points_with_score[(index * 3) + 0])
                score = float(key_points_with_score[(index * 3) + 2])
                key_points.append((x, y, 0, score))

                if score > max_score:
                    max_score = score

            if max_score < self.min_score:
                continue

            poses.append(MoveNetPose(max_score, list_of_vector4D(key_points)))
        return poses

    def release(self):
        pass

    @staticmethod
    def create(config: MoveNetConfig = MoveNetConfig.MoveNet_MultiPose_256x320_FP32) -> "MoveNetPoseEstimator":
        model, weights, height, width = config.value
        return MoveNetPoseEstimator(model, weights, width, height)
