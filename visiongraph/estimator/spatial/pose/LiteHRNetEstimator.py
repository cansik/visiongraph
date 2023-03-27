from enum import Enum
from typing import Tuple, List

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.VisionInferenceEngine import VisionInferenceEngine
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.estimator.spatial.SSDDetector import SSDDetector, SSDConfig
from visiongraph.estimator.spatial.pose.TopDownPoseEstimator import TopDownPoseEstimator
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.ResultUtils import non_maximum_suppression
from visiongraph.util.VectorUtils import list_of_vector4D


class LiteHRNetConfig(Enum):
    LiteHRNet_18_COCO_256x192_FP16 = RepositoryAsset.openVino("litehrnet_18_coco_256x192-fp16")
    LiteHRNet_18_COCO_256x192_FP32 = RepositoryAsset.openVino("litehrnet_18_coco_256x192-fp32")
    LiteHRNet_18_COCO_384x288_FP16 = RepositoryAsset.openVino("litehrnet_18_coco_384x288-fp16")
    LiteHRNet_18_COCO_384x288_FP32 = RepositoryAsset.openVino("litehrnet_18_coco_384x288-fp32")

    LiteHRNet_30_COCO_256x192_FP16 = RepositoryAsset.openVino("litehrnet_30_coco_256x192-fp16")
    LiteHRNet_30_COCO_256x192_FP32 = RepositoryAsset.openVino("litehrnet_30_coco_256x192-fp32")
    LiteHRNet_30_COCO_384x288_FP16 = RepositoryAsset.openVino("litehrnet_30_coco_384x288-fp16")
    LiteHRNet_30_COCO_384x288_FP32 = RepositoryAsset.openVino("litehrnet_30_coco_384x288-fp32")


LITE_HRNET_KEY_POINT_COUNT = 17


class LiteHRNetPoseEstimator(TopDownPoseEstimator[COCOPose]):
    def __init__(self,
                 model: Asset, weights: Asset,
                 human_detector: ObjectDetector = SSDDetector.create(SSDConfig.PersonDetection_0200_256x256_FP32),
                 min_score: float = 0.3, enable_nms: bool = True, iou_threshold: float = 0.4,
                 device: str = "AUTO"):
        super().__init__(human_detector, min_score)

        self.engine = VisionInferenceEngine(model, weights, flip_channels=True, scale=255, device=device)
        self.enable_nms = enable_nms
        self.iou_threshold = iou_threshold

        self.roi_rectified = False

    def setup(self):
        super().setup()
        self.engine.setup()

    def _detect_landmarks(self, image: np.ndarray, roi: np.ndarray, xs: int, ys: int) -> List[COCOPose]:
        h, w = image.shape[:2]
        rh, rw = roi.shape[:2]

        outputs = self.engine.process(roi)
        output = outputs[self.engine.output_names[0]]

        poses: ResultList[COCOPose] = ResultList()
        for raw_result in output:
            key_points: List[Tuple[float, float, float, float]] = []
            max_score = 0.0

            # keypoint
            for index in range(LITE_HRNET_KEY_POINT_COUNT):
                heatmap = raw_result[index]
                hh, hw = heatmap.shape[:2]
                pt = np.unravel_index(np.argmax(heatmap), heatmap.shape)

                x = (pt[1] / hw * rw + xs) / w
                y = (pt[0] / hh * rh + ys) / h

                score = float(heatmap[pt])
                key_points.append((x, y, 0, score))

                if score > max_score:
                    max_score = score

            if max_score < self.min_score:
                continue

            poses.append(COCOPose(max_score, list_of_vector4D(key_points)))

        if self.enable_nms and len(poses) > 1:
            poses = ResultList(non_maximum_suppression(poses, self.min_score, self.iou_threshold))

        return poses

    def release(self):
        super().release()
        self.engine.release()

    @staticmethod
    def create(config: LiteHRNetConfig = LiteHRNetConfig.LiteHRNet_30_COCO_384x288_FP32) -> "LiteHRNetPoseEstimator":
        model, weights = config.value
        return LiteHRNetPoseEstimator(model, weights)
