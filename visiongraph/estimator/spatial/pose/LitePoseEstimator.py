from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util import MathUtils
from visiongraph.util.VectorUtils import list_of_vector4D


class LitePoseEstimatorConfig(Enum):
    LitePose_S_COCO_FP32 = RepositoryAsset.openVino("litepose-auto-s-coco-fp32")
    LitePose_M_COCO_FP32 = RepositoryAsset.openVino("litepose-auto-m-coco-fp32")
    LitePose_L_COCO_FP32 = RepositoryAsset.openVino("litepose-auto-l-coco-fp32")


class LitePoseEstimator(PoseEstimator[COCOPose]):
    def __init__(self, model: Asset, weights: Asset,
                 min_score: float = 0.2, device: str = "AUTO"):
        super().__init__(min_score)

        self.engine = VisionInferenceEngine(model, weights, flip_channels=True, padding=True, device=device)

    def setup(self):
        self.engine.setup()

    def process(self, data: np.ndarray) -> ResultList[COCOPose]:
        output_dict = self.engine.process(data)
        padding_box: BoundingBox2D = output_dict.padding_box
        keypoints = output_dict[self.engine.output_names[1]]
        # pafs = output_dict[self.engine.output_names[0]]

        keypoints = keypoints.reshape(keypoints.shape[1:])
        # pafs = pafs.reshape(pafs.shape[1:])

        poses: ResultList[COCOPose] = ResultList()

        total_score = 0.0
        key_points: List[Tuple[float, float, float, float]] = []

        for heatmap in keypoints:
            w, h = heatmap.shape[:2]
            _, max_val, _, max_indx = cv2.minMaxLoc(heatmap)
            x = max_indx[0] / w
            y = max_indx[1] / h
            score = heatmap[max_indx[1], max_indx[0]]
            key_points.append((
                MathUtils.map_value(x - padding_box.x_min, 0, padding_box.width, 0, 1),
                MathUtils.map_value(y - padding_box.y_min, 0, padding_box.height, 0, 1),
                0.0, float(score)))
            total_score += score

        pose_score = total_score / len(key_points)

        if pose_score < self.min_score:
            return poses

        poses.append(COCOPose(pose_score, list_of_vector4D(key_points)))
        return poses

    def release(self):
        self.engine.release()

    @staticmethod
    def create(config: LitePoseEstimatorConfig
               = LitePoseEstimatorConfig.LitePose_S_COCO_FP32) -> "LitePoseEstimator":
        model, weights = config.value
        return LitePoseEstimator(model, weights)
