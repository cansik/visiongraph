from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.VectorUtils import list_of_vector4D


class KAPAOPoseConfig(Enum):
    KAPAO_N_COCO_640 = RepositoryAsset("kapao_n_coco_640.onnx"), 17
    KAPAO_S_COCO_640 = RepositoryAsset("kapao_s_coco_640.onnx"), 17
    KAPAO_S_COCO_1280 = RepositoryAsset("kapao_s_coco_1280.onnx"), 17
    KAPAO_L_COCO_1280 = RepositoryAsset("kapao_l_coco_1280.onnx"), 17


class KAPAOPoseEstimator(PoseEstimator):
    def __init__(self, *assets: Asset, num_keypoints: int,
                 min_score: float = 0.7, nms_threshold: float = 0.45,
                 kp_min_score: float = 0.1, kp_nms_threshold: float = 0.95,
                 engine: InferenceEngine = InferenceEngine.ONNX):
        super().__init__(min_score)

        self.nms_threshold = nms_threshold

        self.num_keypoints = num_keypoints
        self.kp_min_score = kp_min_score
        self.kp_nms_threshold = kp_nms_threshold

        self.overwrite_tol = 25

        self.engine = InferenceEngineFactory.create(engine, assets,
                                                    flip_channels=True,
                                                    scale=255.0,
                                                    padding=True)
        # set padding color
        self.engine.padding_color = (114, 114, 114)

    def setup(self):
        self.engine.setup()

    def process(self, image: np.ndarray) -> ResultList[COCOPose]:
        h, w = self.engine.first_input_shape[2:]

        output = self.engine.process(image)
        tensor_size = output.image_size
        prediction = output[self.engine.output_names[0]]

        person_dets = self._nms_predictions(prediction, self.min_score, self.nms_threshold, classes=[0])
        kp_dets = self._nms_predictions(prediction, self.kp_min_score, self.kp_nms_threshold,
                                        classes=list(range(1, 1 + self.num_keypoints)))

        _, raw_poses, pose_scores, _, _ = self._post_process_batch(person_dets, kp_dets)

        poses = ResultList()
        for i, raw_pose in enumerate(raw_poses):
            score = pose_scores[i]
            key_points: List[Tuple[float, float, float, float]] = []

            for kp in raw_pose:
                # todo: find out why keypoint conf for last keypoint is not valid
                key_points.append((kp[0] / tensor_size.width, kp[1] / tensor_size.height, 0, 1.0))

            pose = COCOPose(float(score), list_of_vector4D(key_points))
            pose.map_coordinates(output.image_size, Size2D.from_image(image), src_roi=output.padding_box)
            poses.append(pose)

        return poses

    def release(self):
        self.engine.release()

    def _nms_predictions(self, prediction: np.ndarray, threshold: float, nms_threshold: float,
                         classes: Optional[List[int]] = None) -> List[np.ndarray]:
        xc = prediction[..., 4] > threshold

        num_coords = self.num_keypoints * 3
        np_classes = np.array(classes)

        result = []
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            # calculate confidence
            x[:, 5:5 + self.num_keypoints] *= x[:, 4:5]

            kp_conf = x[:, 5:5 + self.num_keypoints]
            j = np.argmax(x[:, 5:5 + self.num_keypoints], 1, keepdims=True)
            confidences = kp_conf[range(kp_conf.shape[0]), j.flatten()]
            kp = x[:, 5 + self.num_keypoints + 1:]
            boxes = self._xywh2xyxy(x[:, :4])

            # Filter by class
            if classes is not None:
                indices, _ = np.where(j == np_classes)
                j = j[indices]
                confidences = confidences[indices]
                kp = kp[indices]
                boxes = boxes[indices]

            nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)
            outs = np.hstack((boxes, confidences.reshape(-1, 1), j, kp))
            outs_filtered = outs[nms_indices]
            result.append(outs_filtered)
        return result

    @staticmethod
    def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _post_process_batch(self, person_dets, kp_dets, origins=None):
        num_coords = self.num_keypoints * 2

        batch_bboxes, batch_poses, batch_scores, batch_ids = [], [], [], []
        n_fused = np.zeros(num_coords // 2)

        if origins is None:  # used only for two-stage inference so set to 0 if None
            origins = [np.array([0, 0, 0]) for _ in range(len(person_dets))]

        # process each image in batch
        for si, (pd, kpd, origin) in enumerate(zip(person_dets, kp_dets, origins)):
            nd = pd.shape[0]
            nkp = kpd.shape[0]

            if nd:
                scores = pd[:, 4]  # person detection score
                bboxes = pd[:, :4].round()
                poses = pd[:, -num_coords:]
                poses = poses.reshape((nd, -num_coords, 2))
                poses = np.concatenate((poses, np.zeros((nd, poses.shape[1], 1))), axis=-1)

                if nkp:
                    mask = scores > self.kp_min_score
                    poses_mask = poses[mask]

                    if len(poses_mask):
                        kpd = kpd[:, :6]

                        for x1, y1, x2, y2, conf, cls in kpd:
                            x, y = np.mean((x1, x2)), np.mean((y1, y2))
                            pose_kps = poses_mask[:, int(cls - 1)]
                            dist = np.linalg.norm(pose_kps[:, :2] - np.array([[x, y]]), axis=-1)
                            kp_match = np.argmin(dist)
                            if conf > pose_kps[kp_match, 2] and dist[kp_match] < self.overwrite_tol:
                                pose_kps[kp_match] = [x, y, conf]
                        poses[mask] = poses_mask

                poses = [p + origin for p in poses]

                batch_bboxes.extend(bboxes)
                batch_poses.extend(poses)
                batch_scores.extend(scores)

        return batch_bboxes, batch_poses, batch_scores, batch_ids, n_fused

    @staticmethod
    def create(config: KAPAOPoseConfig = KAPAOPoseConfig.KAPAO_S_COCO_640) -> "KAPAOPoseEstimator":
        model, num_keypoints = config.value
        return KAPAOPoseEstimator(model, num_keypoints=num_keypoints)
