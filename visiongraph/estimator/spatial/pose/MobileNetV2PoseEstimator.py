from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.openvino.OpenVinoEngine import OpenVinoEngine
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.types.InputShapeOrder import InputShapeOrder
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOOpenPose import COCOOpenPose, COCO_OPEN_POSE_KEYPOINT_COUNT
from visiongraph.util.VectorUtils import list_of_vector4D

_MAP_INDEX = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
              [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]

_POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
               [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

_COLORS = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
           [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
           [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]


class MobileNetV2PoseEstimatorConfig(Enum):
    """
    Enumeration of MobileNetV2 Pose Estimator configurations.
    Each configuration corresponds to a specific model and weight variant.
    """
    MNV2PE_0_5_224_FP16 = RepositoryAsset.openVino("mobilenet_v2_pose_0.5_224-fp16")
    MNV2PE_0_5_224_FP32 = RepositoryAsset.openVino("mobilenet_v2_pose_0.5_224-fp32")
    MNV2PE_0_5_224_QUANT_FP16 = RepositoryAsset.openVino("mobilenet_v2_pose_0.5_224_quant-fp16")
    MNV2PE_0_5_224_QUANT_FP32 = RepositoryAsset.openVino("mobilenet_v2_pose_0.5_224_quant-fp32")
    MNV2PE_0_75_224_FP16 = RepositoryAsset.openVino("mobilenet_v2_pose_0.75_224-fp16")
    MNV2PE_0_75_224_FP32 = RepositoryAsset.openVino("mobilenet_v2_pose_0.75_224-fp32")
    MNV2PE_1_0_224_FP16 = RepositoryAsset.openVino("mobilenet_v2_pose_1.0_224-fp16")
    MNV2PE_1_0_224_FP32 = RepositoryAsset.openVino("mobilenet_v2_pose_1.0_224-fp32")
    MNV2PE_1_4_224_FP16 = RepositoryAsset.openVino("mobilenet_v2_pose_1.4_224-fp16")
    MNV2PE_1_4_224_FP32 = RepositoryAsset.openVino("mobilenet_v2_pose_1.4_224-fp32")
    MNV2PE_1_4_224_QUANT_FP16 = RepositoryAsset.openVino("mobilenet_v2_pose_1.4_224_quant-fp16")
    MNV2PE_1_4_224_QUANT_FP32 = RepositoryAsset.openVino("mobilenet_v2_pose_1.4_224_quant-fp32")


class MobileNetV2PoseEstimator(PoseEstimator[COCOOpenPose]):
    """
    Pose Estimator using MobileNetV2 architecture.

    :param model: The model asset to be used.
    :param weights: The weights asset for the model.
    :param min_score: Minimum score threshold for detected keypoints. Defaults to 0.2.
    :param device: Device type for running the model. Defaults to "AUTO".
    """

    def __init__(self, model: Asset, weights: Asset,
                 min_score: float = 0.2, device: str = "AUTO"):
        super().__init__(min_score)

        self.engine = OpenVinoEngine(model, weights, flip_channels=True, device=device)
        self.engine.order = InputShapeOrder.NWHC
        self.threshold = 0.1

    def setup(self):
        """
        Prepares the engine for inference.
        """
        self.engine.setup()

    def process(self, data: np.ndarray) -> ResultList[COCOOpenPose]:
        """
        Processes input data and returns detected poses.

        :param data: Input data for pose estimation.

        :return: A list of detected poses with their corresponding keypoints.
        """
        output_dict = self.engine.process(data)
        outputs_nhwc = output_dict[self.engine.output_names[0]]

        # transpose data to nchw from nhwc
        outputs = np.transpose(outputs_nhwc, (0, 3, 1, 2))

        h, w = outputs.shape[2:]

        detected_keypoints = []
        keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for part in range(COCO_OPEN_POSE_KEYPOINT_COUNT):
            probability_map = outputs[0, part, :, :]
            keypoints = self._get_keypoints(probability_map, self.threshold)
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = self._get_valid_pairs(outputs, w, h, detected_keypoints)
        personwise_keypoints = self._get_personwise_keypoints(valid_pairs, invalid_pairs, keypoints_list)

        poses: ResultList[COCOOpenPose] = ResultList()
        for person in personwise_keypoints:
            total_score = 0.0
            key_points: List[Tuple[float, float, float, float]] = []

            for i in range(COCO_OPEN_POSE_KEYPOINT_COUNT):
                index = int(person[i])

                if index == -1:
                    key_points.append((0.0, 0.0, 0.0, 0.0))
                    continue

                kp = keypoints_list[index]
                x = kp[0] / w
                y = kp[1] / h
                score = kp[2]

                total_score += score
                key_points.append((x, y, 0, score))

            pose_score = total_score / COCO_OPEN_POSE_KEYPOINT_COUNT

            if pose_score < self.min_score:
                continue

            poses.append(COCOOpenPose(pose_score, list_of_vector4D(key_points)))

        return poses

    def release(self):
        """
        Releases the resources used by the engine.
        """
        self.engine.release()

    @staticmethod
    def _get_keypoints(probability_map: np.ndarray, threshold: float = 0.1):
        """
        Extracts keypoints from the probability map.

        :param probability_map: The probability map from which to extract keypoints.
        :param threshold: Threshold to filter weak keypoints. Defaults to 0.1.

        :return: List of detected keypoints with their (x, y) coordinates and confidence scores.
        """
        map_smooth = cv2.GaussianBlur(probability_map, (3, 3), 0, 0)
        map_mask = np.uint8(map_smooth > threshold)
        keypoints = []
        contours = None
        contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            blobMask = np.zeros(map_mask.shape)
            blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
            maskedProbMap = map_smooth * blobMask
            _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
            keypoints.append(maxLoc + (probability_map[maxLoc[1], maxLoc[0]],))

        return keypoints

    @staticmethod
    def _get_valid_pairs(outputs, w, h, detected_keypoints):
        """
        Determines valid and invalid pairs of keypoints.

        :param outputs: The model's output.
        :param w: Width of the output.
        :param h: Height of the output.
        :param detected_keypoints: List of detected keypoints.

        :return: A tuple containing valid pairs of keypoints and indices of invalid pairs.
        """
        valid_pairs = []
        invalid_pairs = []
        n_interp_samples = 10
        paf_score_th = 0.1
        conf_th = 0.7

        for k in range(len(_MAP_INDEX)):
            pafA = outputs[0, _MAP_INDEX[k][0], :, :]
            pafB = outputs[0, _MAP_INDEX[k][1], :, :]
            pafA = cv2.resize(pafA, (w, h))
            pafB = cv2.resize(pafB, (w, h))

            candA = detected_keypoints[_POSE_PAIRS[k][0]]
            candB = detected_keypoints[_POSE_PAIRS[k][1]]
            nA = len(candA)
            nB = len(candB)

            if nA != 0 and nB != 0:
                valid_pair = np.zeros((0, 3))
                for i in range(nA):
                    max_j = -1
                    maxScore = -1
                    found = 0
                    for j in range(nB):
                        d_ij = np.subtract(candB[j][:2], candA[i][:2])
                        norm = np.linalg.norm(d_ij)
                        if norm:
                            d_ij = d_ij / norm
                        else:
                            continue
                        interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                                np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                        paf_interp = []
                        for k in range(len(interp_coord)):
                            paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                               pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                        paf_scores = np.dot(paf_interp, d_ij)
                        avg_paf_score = sum(paf_scores) / len(paf_scores)

                        if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                            if avg_paf_score > maxScore:
                                max_j = j
                                maxScore = avg_paf_score
                                found = 1
                    if found:
                        valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

                valid_pairs.append(valid_pair)
            else:
                invalid_pairs.append(k)
                valid_pairs.append([])
        return valid_pairs, invalid_pairs

    @staticmethod
    def _get_personwise_keypoints(valid_pairs, invalid_pairs, keypoints_list):
        """
        Aggregates keypoints for each detected person based on valid pairs.

        :param valid_pairs: List of valid pairs of keypoints.
        :param invalid_pairs: List of indices for invalid pairs.
        :param keypoints_list: Array of keypoints.

        :return: Array of personwise keypoints, where each row corresponds to a person.
        """
        personwiseKeypoints = -1 * np.ones((0, 19))

        for k in range(len(_MAP_INDEX)):
            if k not in invalid_pairs:
                partAs = valid_pairs[k][:, 0]
                partBs = valid_pairs[k][:, 1]
                indexA, indexB = np.array(_POSE_PAIRS[k])

                for i in range(len(valid_pairs[k])):
                    found = 0
                    person_idx = -1
                    for j in range(len(personwiseKeypoints)):
                        if personwiseKeypoints[j][indexA] == partAs[i]:
                            person_idx = j
                            found = 1
                            break

                    if found:
                        personwiseKeypoints[person_idx][indexB] = partBs[i]
                        personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + \
                            valid_pairs[k][i][2]

                    elif not found and k < 17:
                        row = -1 * np.ones(19)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                        personwiseKeypoints = np.vstack([personwiseKeypoints, row])
        return personwiseKeypoints

    @staticmethod
    def create(config: MobileNetV2PoseEstimatorConfig
               = MobileNetV2PoseEstimatorConfig.MNV2PE_1_4_224_FP32) -> "MobileNetV2PoseEstimator":
        """
        Factory method to create a MobileNetV2PoseEstimator instance.

        :param config: The configuration for the pose estimator. Defaults to MNV2PE_1_4_224_FP32.

        :return: An instance of the MobileNetV2PoseEstimator.
        """
        model, weights = config.value
        return MobileNetV2PoseEstimator(model, weights)
