from enum import Enum
from typing import List, Tuple

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.ResultUtils import non_maximum_suppression
from visiongraph.util.VectorUtils import list_of_vector4D


class MoveNetConfig(Enum):
    MoveNet_MultiPose_192x192_FP32 = RepositoryAsset.openVino("movenet-multipose-192x192-fp32")
    MoveNet_MultiPose_192x256_FP32 = RepositoryAsset.openVino("movenet-multipose-192x256-fp32")
    MoveNet_MultiPose_256x256_FP32 = RepositoryAsset.openVino("movenet-multipose-256x256-fp32")
    MoveNet_MultiPose_256x320_FP32 = RepositoryAsset.openVino("movenet-multipose-256x320-fp32")
    MoveNet_MultiPose_320x320_FP32 = RepositoryAsset.openVino("movenet-multipose-320x320-fp32")
    MoveNet_MultiPose_480x640_FP32 = RepositoryAsset.openVino("movenet-multipose-480x640-fp32")
    MoveNet_MultiPose_736x1280_FP32 = RepositoryAsset.openVino("movenet-multipose-736x1280-fp32")
    MoveNet_MultiPose_1280x1920_FP32 = RepositoryAsset.openVino("movenet-multipose-1280x1920-fp32")


MOVE_NET_KEY_POINT_COUNT = 17


class MoveNetPoseEstimator(PoseEstimator[COCOPose]):
    def __init__(self, model: Asset, weights: Asset,
                 min_score: float = 0.3, enable_nms: bool = True, iou_threshold: float = 0.4,
                 device: str = "CPU"):
        super().__init__(min_score)

        self.engine = VisionInferenceEngine(model, weights, device=device)
        self.enable_nms = enable_nms
        self.iou_threshold = iou_threshold

    def setup(self):
        self.engine.setup()

    def process(self, data: np.ndarray) -> ResultList[COCOPose]:
        outputs = self.engine.process(data)
        output = outputs[self.engine.output_names[0]]

        key_points_with_scores = output[0]
        key_points_with_scores = np.squeeze(key_points_with_scores)

        poses: ResultList[COCOPose] = ResultList()
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

            poses.append(COCOPose(max_score, list_of_vector4D(key_points)))

        if self.enable_nms:
            poses = ResultList(non_maximum_suppression(poses, self.min_score, self.iou_threshold))

        return poses

    def release(self):
        pass

    @staticmethod
    def create(config: MoveNetConfig = MoveNetConfig.MoveNet_MultiPose_256x320_FP32) -> "MoveNetPoseEstimator":
        model, weights = config.value
        return MoveNetPoseEstimator(model, weights)
