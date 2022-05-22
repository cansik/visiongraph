from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import numpy as np
import onnxruntime as rt
from scipy.special import softmax

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.estimator.spatial.SSDDetector import SSDDetector, SSDConfig
from visiongraph.estimator.spatial.pose.TopDownPoseEstimator import TopDownPoseEstimator, OutputType
from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.MobileHumanPose import MobileHumanPose
from visiongraph.util import VectorUtils

MOBILE_HUMAN_POSE_JOINT_NUM = 21


@dataclass
class _RawMobileHumanPoseResult:
    pose_2d: np.ndarray
    pose_3d: np.ndarray
    scores: np.ndarray
    person_heatmap: Optional[np.ndarray] = None


class MobileHumanPoseEstimator(TopDownPoseEstimator[MobileHumanPose]):
    def __init__(self,
                 human_detector: ObjectDetector = SSDDetector.create(SSDConfig.PersonDetection_0201_384x384_FP32),
                 model: Asset = RepositoryAsset("mobile_human_pose_working_well_256x256.onnx"),
                 intrinsics: Optional[CameraIntrinsics] = None,
                 abs_depth: float = 1.0,
                 min_score: float = 0.5):
        super().__init__(human_detector, min_score)

        self.model = model
        self.intrinsics = intrinsics
        self.abs_depth = abs_depth

        self.session: Optional[rt.InferenceSession] = None
        self.session_options = rt.SessionOptions()

        self.input_name: str = ""
        self.channels: Optional[int] = None
        self.input_width: Optional[int] = None
        self.input_height: Optional[int] = None

        self.output_names: Optional[List[str]] = None
        self.output_depth: Optional[int] = None
        self.output_width: Optional[int] = None
        self.output_height: Optional[int] = None

    def setup(self):
        super().setup()
        self.session = rt.InferenceSession(self.model.path,
                                           providers=["CUDAExecutionProvider",
                                                      "OpenVINOExecutionProvider",
                                                      "CPUExecutionProvider"],
                                           sess_options=self.session_options)

        # read input infos
        self.input_name = self.session.get_inputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        self.channels = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        # read output infos
        model_outputs = self.session.get_outputs()

        self.output_names = []
        self.output_names.append(model_outputs[0].name)

        output_shape = model_outputs[0].shape
        self.output_depth = output_shape[1] // MOBILE_HUMAN_POSE_JOINT_NUM
        self.output_height = output_shape[2]
        self.output_width = output_shape[3]

    def _pre_landmark(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _detect_landmarks(self, image: np.ndarray, roi: np.ndarray, xs: int, ys: int) -> ResultList[OutputType]:
        h, w = image.shape[:2]
        rh, rw = roi.shape[:2]

        # prepare input
        img_input = cv2.resize(roi, (self.input_width, self.input_height))
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]
        input_tensor = img_input.astype(np.float32)

        # process
        output_tensor = self.session.run(self.output_names, {self.input_name: input_tensor})[0]
        output = np.squeeze(output_tensor)

        # post process
        raw_result = self._post_process(output)

        # create pose
        landmarks: List[Tuple[float, float, float, float]] = []
        max_score = 0.0
        for i, point in enumerate(raw_result.pose_3d):
            score = raw_result.scores[i]
            x = (point[0] * rw + xs) / w
            y = (point[1] * rh + ys) / h
            z = point[2]
            landmarks.append((x, y, z, float(score)))

            if max_score < score:
                max_score = float(score)

        return ResultList([MobileHumanPose(max_score, VectorUtils.list_of_vector4D(landmarks))])

    def _post_process(self, output: np.ndarray) -> _RawMobileHumanPoseResult:
        heatmaps = output.reshape((-1, MOBILE_HUMAN_POSE_JOINT_NUM,
                                   self.output_depth * self.output_height * self.output_width))
        heatmaps = softmax(heatmaps, 2)

        scores = np.squeeze(np.max(heatmaps, 2))  # Ref: https://github.com/mks0601/3DMPPE_POSENET_RELEASE/issues/47

        heatmaps = heatmaps.reshape((-1, MOBILE_HUMAN_POSE_JOINT_NUM,
                                     self.output_depth, self.output_height, self.output_width))

        accu_x = heatmaps.sum(axis=(2, 3))
        accu_y = heatmaps.sum(axis=(2, 4))
        accu_z = heatmaps.sum(axis=(3, 4))

        accu_x = accu_x * np.arange(self.output_width, dtype=np.float32)
        accu_y = accu_y * np.arange(self.output_height, dtype=np.float32)
        accu_z = accu_z * np.arange(self.output_depth, dtype=np.float32)

        accu_x = accu_x.sum(axis=2, keepdims=True)
        accu_y = accu_y.sum(axis=2, keepdims=True)
        accu_z = accu_z.sum(axis=2, keepdims=True)

        scores2 = []
        for i in range(MOBILE_HUMAN_POSE_JOINT_NUM):
            scores2.append(heatmaps.sum(axis=2)[0, i, int(accu_y[0, i, 0]), int(accu_x[0, i, 0])])

        accu_x = accu_x / self.output_width
        accu_y = accu_y / self.output_height
        accu_z = accu_z / self.output_depth * 2 - 1

        coord_out = np.squeeze(np.concatenate((accu_x, accu_y, accu_z), axis=2))

        pose_2d = coord_out[:, :2]
        pose_2d[:, 0] = pose_2d[:, 0]
        pose_2d[:, 1] = pose_2d[:, 1]

        joint_depth = coord_out[:, 2] * 1000 + self.abs_depth

        # todo: use intrinsics to calculate pixel 2 cam
        pose_3d = np.concatenate((pose_2d[:, 0][:, None], pose_2d[:, 1][:, None], joint_depth[:, None]), 1)
        # pose_3d = pixel2cam(pose_2d, joint_depth, self.focal_length, self.principal_points)

        # Calculate the joint heatmap
        # person_heatmap = cv2.resize(np.sqrt(heatmaps.sum(axis=(1, 2))[0, :, :]), (self.img_width, self.img_height))

        return _RawMobileHumanPoseResult(pose_2d, pose_3d, scores)

    def release(self):
        super().release()
        self.session = None
