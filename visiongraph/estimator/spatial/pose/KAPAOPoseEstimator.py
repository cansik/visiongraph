from enum import Enum
from typing import List, Optional

import numpy as np

from visiongraph.data.Asset import Asset
from visiongraph.data.RepositoryAsset import RepositoryAsset
from visiongraph.data.labels.COCO import COCO_80_LABELS
from visiongraph.estimator.engine.InferenceEngineFactory import InferenceEngine, InferenceEngineFactory
from visiongraph.estimator.spatial.ObjectDetector import ObjectDetector
from visiongraph.estimator.spatial.YOLOv5Detector import YOLOv5Detector
from visiongraph.estimator.spatial.pose.PoseEstimator import PoseEstimator
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.geometry.Size2D import Size2D
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.ObjectDetectionResult import ObjectDetectionResult
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util.ResultUtils import non_maximum_suppression


class KAPAOPoseConfig(Enum):
    KAPAO_S_COCO_1280 = RepositoryAsset("kapao_s_coco_1280.onnx"), 17


class KAPAOPoseEstimator(PoseEstimator):
    def __init__(self, *assets: Asset, num_keypoints: int,
                 min_score: float = 0.7, nms_threshold: float = 0.45,
                 engine: InferenceEngine = InferenceEngine.ONNX):
        super().__init__(min_score)

        self.num_keypoints = num_keypoints
        self.nms_threshold = nms_threshold

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
        prediction = output[self.engine.output_names[0]]

        xc = prediction[..., 4] > self.min_score

        num_coords = self.num_keypoints * 3

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            # calculate confidence
            x[:, 5:-num_coords] *= x[:, 4:5]

            kp_conf = x[:, 5:5 + self.num_keypoints]
            j = np.argmax(x[:, 5:5 + self.num_keypoints], 1, keepdims=True)
            conf = np.take(kp_conf, j)
            kp = x[:, 5 + self.num_keypoints + 1:]

            det_bbox = x[0:4]

            # process bounding box
            wh = det_bbox[2:]
            xy = det_bbox[:2]
            xy -= wh * 0.5
            bbox = BoundingBox2D(xy[0], xy[1], wh[0], wh[1]).scale(1 / w, 1 / h)

            # filter detection min score
            detections = detections[np.where(detections[:, 4] > self.min_score)]

    def release(self):
        self.engine.release()

    def _post_process_batch(self, data, imgs, paths, shapes, person_dets, kp_dets,
                            two_stage=False, pad=0, device='cpu', model=None, origins=None):

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
                path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]
                img_id = int(osp.splitext(osp.split(path)[-1])[0]) if path else si

                scores = pd[:, 4].cpu().numpy()  # person detection score
                bboxes = scale_coords(imgs[si].shape[1:], pd[:, :4], shape).round().cpu().numpy()
                poses = scale_coords(imgs[si].shape[1:], pd[:, -num_coords:], shape).cpu().numpy()
                poses = poses.reshape((nd, -num_coords, 2))
                poses = np.concatenate((poses, np.zeros((nd, poses.shape[1], 1))), axis=-1)

                if data['use_kp_dets'] and nkp:
                    mask = scores > data['conf_thres_kp_person']
                    poses_mask = poses[mask]

                    if len(poses_mask):
                        kpd[:, :4] = scale_coords(imgs[si].shape[1:], kpd[:, :4], shape)
                        kpd = kpd[:, :6].cpu()

                        for x1, y1, x2, y2, conf, cls in kpd:
                            x, y = np.mean((x1, x2)), np.mean((y1, y2))
                            pose_kps = poses_mask[:, int(cls - 1)]
                            dist = np.linalg.norm(pose_kps[:, :2] - np.array([[x, y]]), axis=-1)
                            kp_match = np.argmin(dist)
                            if conf > pose_kps[kp_match, 2] and dist[kp_match] < data['overwrite_tol']:
                                pose_kps[kp_match] = [x, y, conf]
                                if data['count_fused']:
                                    n_fused[int(cls - 1)] += 1
                        poses[mask] = poses_mask

                poses = [p + origin for p in poses]

                batch_bboxes.extend(bboxes)
                batch_poses.extend(poses)
                batch_scores.extend(scores)
                batch_ids.extend([img_id] * len(scores))

        return batch_bboxes, batch_poses, batch_scores, batch_ids, n_fused

    @staticmethod
    def create(config: KAPAOPoseConfig = KAPAOPoseConfig.KAPAO_S_COCO_1280) -> "KAPAOPoseEstimator":
        model, num_keypoints = config.value
        return KAPAOPoseEstimator(model, num_keypoints=num_keypoints)
