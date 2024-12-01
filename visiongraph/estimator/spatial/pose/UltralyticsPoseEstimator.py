from enum import Enum
from typing import List, Tuple, Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.ResultUtils import non_maximum_suppression
from visiongraph.util.VectorUtils import list_of_vector4D


class UltralyticsPoseConfig(Enum):
    """
    Configuration options for the Ultralytics pose estimation models.
    """
    YOLOv8_N_640 = RepositoryAsset("yolov8n-pose-8-1.onnx"), InferenceEngine.ONNX, 17
    YOLOv8_S_640 = RepositoryAsset("yolov8s-pose-8-1.onnx"), InferenceEngine.ONNX, 17
    YOLOv8_M_640 = RepositoryAsset("yolov8m-pose-8-1.onnx"), InferenceEngine.ONNX, 17
    YOLOv8_L_640 = RepositoryAsset("yolov8l-pose-8-1.onnx"), InferenceEngine.ONNX, 17
    YOLOv8_X_640 = RepositoryAsset("yolov8x-pose-8-1.onnx"), InferenceEngine.ONNX, 17

    YOLOv8_N_640_INT8 = *RepositoryAsset.openVino("yolov8n-pose-8-1-INT8"), InferenceEngine.OpenVINO2, 17
    YOLOv8_S_640_INT8 = *RepositoryAsset.openVino("yolov8s-pose-8-1-INT8"), InferenceEngine.OpenVINO2, 17

    YOLOv11_N_640 = RepositoryAsset("yolo11n-pose.onnx"), InferenceEngine.ONNX, 17
    YOLOv11_S_640 = RepositoryAsset("yolo11s-pose.onnx"), InferenceEngine.ONNX, 17


class UltralyticsPoseEstimator(PoseEstimator):
    """
    A class for performing pose estimation using the Ultralytics Pose models (YOLOv8, YOLO11, ...).

    Please be aware that Ultralytics publishes their models under AGPLv3 license.

    Inherits from:
        PoseEstimator: Base class for pose estimation algorithms.
    """

    def __init__(self, *assets: Asset, num_keypoints: int,
                 min_score: float = 0.7, nms: bool = True, nms_threshold: float = 0.5,
                 nms_eta: Optional[float] = None, nms_top_k: Optional[int] = None,
                 engine: InferenceEngine = InferenceEngine.OpenVINO2):
        """
        Initializes the UltralyticsPoseEstimator with the given parameters.

        :param assets: Assets required for the pose estimation model.
        :param num_keypoints: The number of keypoints to detect.
        :param min_score: Minimum score threshold for keypoints detection.
        :param nms: Whether to apply non-maximum suppression.
        :param nms_threshold: Threshold for non-maximum suppression.
        :param nms_eta: Eta parameter for non-maximum suppression.
        :param nms_top_k: Maximum number of detections to keep after non-maximum suppression.
        :param engine: Inference engine to run the pose estimation model.
        """
        super().__init__(min_score)

        self.nms_threshold: float = nms_threshold
        self.nms: bool = nms
        self.nms_eta = nms_eta
        self.nms_top_k = nms_top_k

        self.num_keypoints = num_keypoints

        self.engine = InferenceEngineFactory.create(engine, assets,
                                                    flip_channels=True,
                                                    scale=255.0,
                                                    padding=True)
        # set padding color
        self.engine.padding_color = (114, 114, 114)

    def setup(self):
        """
        Prepares the inference engine for processing.
        """
        self.engine.setup()

    def process(self, image: np.ndarray) -> ResultList[COCOPose]:
        """
        Processes an image and performs pose estimation.

        :param image: Input image in which poses are to be detected.

        :return: A list of detected poses represented as COCOPose objects.
        """
        h, w = self.engine.first_input_shape[2:]

        output = self.engine.process(image)
        tensor_size = output.image_size
        predictions = output[self.engine.output_names[0]]

        prediction = predictions[0]
        prediction = prediction.T

        # result:
        # detected boxes in format [x1, y1, w, h, score] and kpt - 17 keypoints in format [x1, y1, score1]

        # filter numpy array using score
        prediction = prediction[prediction[:, 4] >= self.min_score]

        poses = ResultList()
        for i, raw_pose in enumerate(prediction):
            x1, y1, bw, bh, score = raw_pose[:5]
            raw_key_points = raw_pose[5:].reshape((self.num_keypoints, 3))

            box = BoundingBox2D(x1 / tensor_size.width, y1 / tensor_size.height,
                                bw / tensor_size.width, bh / tensor_size.height)

            key_points: List[Tuple[float, float, float, float]] = []
            for kp in raw_key_points:
                key_points.append((kp[0] / tensor_size.width, kp[1] / tensor_size.height, 0, float(kp[2])))

            pose = COCOPose(float(score), list_of_vector4D(key_points), box)
            pose.map_coordinates(output.image_size, Size2D.from_image(image), src_roi=output.padding_box)
            poses.append(pose)

        if self.nms:
            poses = ResultList(non_maximum_suppression(poses, self.min_score, self.nms_threshold,
                                                       self.nms_eta, self.nms_top_k))

        return poses

    def release(self):
        """
        Releases the resources held by the inference engine.
        """
        self.engine.release()

    @staticmethod
    def create(config: UltralyticsPoseConfig = UltralyticsPoseConfig.YOLOv8_S_640) -> "UltralyticsPoseEstimator":
        """
        Creates an instance of UltralyticsPoseEstimator based on the provided configuration.

        :param config: Configuration for the Ultralytics pose estimator.

        :return: An instance of the UltralyticsPoseEstimator.
        """
        num_args = len(config.value) - 2

        assets = config.value[:num_args]
        engine = config.value[-2]
        num_keypoints = config.value[-1]

        if type(assets) is not tuple:
            assets = (assets,)

        return UltralyticsPoseEstimator(*assets, num_keypoints=num_keypoints, engine=engine)
