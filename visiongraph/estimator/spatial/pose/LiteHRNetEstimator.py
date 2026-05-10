from enum import Enum
from typing import Tuple, List, Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.estimator.spatial.SSDDetector import SSDDetector, SSDConfig
from visiongraph.estimator.spatial.pose.TopDownPoseEstimator import TopDownPoseEstimator
from visiongraph.model.NMSOptions import NMSOptions
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.ResultUtils import non_maximum_suppression_from_options
from visiongraph.util.VectorUtils import list_of_vector4D


_MODEL_METADATA = AssetMetadata.from_values(
    source_name="Lite-HRNet pose estimation",
    source_url="https://github.com/HRNet/Lite-HRNet",
    license_name="Apache License 2.0",
    license_url="https://github.com/HRNet/Lite-HRNet/blob/hrnet/LICENSE",
)


class LiteHRNetConfig(Enum):
    """
    Enumeration for LiteHRNet model configurations with corresponding assets.
    """

    LiteHRNet_18_COCO_256x192_FP16 = RepositoryAsset.openVino(
        "litehrnet_18_coco_256x192-fp16", metadata=_MODEL_METADATA
    )
    LiteHRNet_18_COCO_256x192_FP32 = RepositoryAsset.openVino(
        "litehrnet_18_coco_256x192-fp32", metadata=_MODEL_METADATA
    )
    LiteHRNet_18_COCO_384x288_FP16 = RepositoryAsset.openVino(
        "litehrnet_18_coco_384x288-fp16", metadata=_MODEL_METADATA
    )
    LiteHRNet_18_COCO_384x288_FP32 = RepositoryAsset.openVino(
        "litehrnet_18_coco_384x288-fp32", metadata=_MODEL_METADATA
    )

    LiteHRNet_30_COCO_256x192_FP16 = RepositoryAsset.openVino(
        "litehrnet_30_coco_256x192-fp16", metadata=_MODEL_METADATA
    )
    LiteHRNet_30_COCO_256x192_FP32 = RepositoryAsset.openVino(
        "litehrnet_30_coco_256x192-fp32", metadata=_MODEL_METADATA
    )
    LiteHRNet_30_COCO_384x288_FP16 = RepositoryAsset.openVino(
        "litehrnet_30_coco_384x288-fp16", metadata=_MODEL_METADATA
    )
    LiteHRNet_30_COCO_384x288_FP32 = RepositoryAsset.openVino(
        "litehrnet_30_coco_384x288-fp32", metadata=_MODEL_METADATA
    )


LITE_HRNET_KEY_POINT_COUNT = 17
"""
Constant representing the number of key points for the LiteHRNet model.
"""


class LiteHRNetPoseEstimator(TopDownPoseEstimator[COCOPose]):
    """
    A pose estimator based on the LiteHRNet architecture for detecting human poses.
    """

    def __init__(
        self,
        model: Asset,
        weights: Asset,
        human_detector: ObjectDetector = SSDDetector.create(SSDConfig.PersonDetection_0200_256x256_FP32),
        min_score: float = 0.3,
        nms_options: Optional[NMSOptions] = None,
        device: str = "AUTO",
    ):
        """
        Initializes the LiteHRNetPoseEstimator with the specified model, weights, and parameters.

        :param model: The model asset for pose estimation.
        :param weights: The weights asset for pose estimation.
        :param human_detector: An object detector for detecting humans. Defaults to SSDDetector.
        :param min_score: The minimum score for detected poses. Defaults to 0.3.
        :param nms_options: Non-Maximum-Suppression options.
        :param device: The device to use for inference. Defaults to "AUTO".
        """
        super().__init__(human_detector, min_score)

        self.engine = OpenVinoEngine(model, weights, flip_channels=True, scale=255, device=device)
        self.nms_options = nms_options or NMSOptions()

        self.roi_rectified = False

    def setup(self):
        """
        Sets up the pose estimator and the underlying engine.
        """
        super().setup()
        self.engine.setup()

    def _detect_landmarks(self, image: np.ndarray, roi: np.ndarray, xs: int, ys: int) -> List[COCOPose]:
        """
        Detects key points in the provided region of interest (ROI) of the image.

        :param image: The input image in which to detect poses.
        :param roi: The region of interest for detection.
        :param xs: The x-coordinate offset of the ROI in the image.
        :param ys: The y-coordinate offset of the ROI in the image.

        :return: A list of detected poses, each represented by COCOPose objects.
        """
        h, w = image.shape[:2]
        rh, rw = roi.shape[:2]

        outputs = self.engine.process(roi)
        output = outputs[self.engine.output_names[0]]

        poses: ResultList[COCOPose] = ResultList()
        for raw_result in output:
            key_points: List[Tuple[float, float, float, float]] = []
            max_score = 0.0

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

        if self.nms_options.enabled and len(poses) > 1:
            poses = ResultList(non_maximum_suppression_from_options(poses, self.nms_options))

        return poses

    def release(self):
        """
        Releases resources used by the pose estimator and the engine.
        """
        super().release()
        self.engine.release()

    @staticmethod
    def create(config: LiteHRNetConfig = LiteHRNetConfig.LiteHRNet_30_COCO_384x288_FP32) -> "LiteHRNetPoseEstimator":
        """
        Creates an instance of LiteHRNetPoseEstimator based on the specified configuration.

        :param config: The configuration to use for creating the estimator. Defaults to LiteHRNet_30_COCO_384x288_FP32.

        :return: An instance of the LiteHRNetPoseEstimator.
        """
        model, weights = config.value
        return LiteHRNetPoseEstimator(model, weights)
