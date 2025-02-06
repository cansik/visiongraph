import unittest

import cv2
import numpy as np
from vector import VectorNumpy4D

from visiongraph import vg
from visiongraph.estimator.embedding.LandmarkEmbedder import LandmarkEmbedder
from visiongraph.result.ResultList import ResultList
from visiongraph.result.spatial.pose.COCOPose import COCOPose
from visiongraph.util import PoseUtils
from visiongraph.util.VectorUtils import list_of_vector4D


class PoseEstimationTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.PoseEstimator):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        model.setup()
        model.process(image)
        model.release()

    def test_aepose_estimator_fp32(self):
        self._test_model(vg.AEPoseEstimator.create(vg.AEPoseConfig.EfficientHRNet_288_FP32))

    def test_aepose_estimator_fp16(self):
        self._test_model(vg.AEPoseEstimator.create(vg.AEPoseConfig.EfficientHRNet_288_FP16))

    def test_effecient_pose_estimator_fp16(self):
        self._test_model(vg.EfficientPoseEstimator.create(vg.EfficientPoseEstimatorConfig.EFFICIENT_POSE_I_LITE_FP16))

    def test_effecient_pose_estimator_fp32(self):
        self._test_model(vg.EfficientPoseEstimator.create(vg.EfficientPoseEstimatorConfig.EFFICIENT_POSE_I_LITE_FP32))

    def test_lite_hrnet_estimator_fp16(self):
        self._test_model(vg.LiteHRNetPoseEstimator.create(vg.LiteHRNetConfig.LiteHRNet_18_COCO_256x192_FP16))

    def test_lite_hrnet_estimator_fp32(self):
        self._test_model(vg.LiteHRNetPoseEstimator.create(vg.LiteHRNetConfig.LiteHRNet_18_COCO_256x192_FP32))

    def test_lite_pose_estimator_fp32(self):
        self._test_model(vg.LitePoseEstimator.create(vg.LitePoseEstimatorConfig.LitePose_S_COCO_FP32))

    def test_mediapipe_pose_estimator(self):
        self._test_model(vg.MediaPipePoseEstimator.create(vg.MediaPipePoseConfig.Light))

    def test_legacy_mediapipe_pose_estimator(self):
        self._test_model(vg.MediaPipePoseEstimatorLegacy.create(vg.PoseModelComplexity.Light))

    def test_legacy_mediapipe_holistic_estimator(self):
        self._test_model(vg.MediaPipeHolisticEstimator.create(vg.PoseModelComplexity.Light))

    def test_mobile_human_pose_estimator(self):
        self._test_model(vg.MobileHumanPoseEstimator())

    def test_mobilenetv2_pose_estimator_fp16(self):
        self._test_model(vg.MobileNetV2PoseEstimator.create(vg.MobileNetV2PoseEstimatorConfig.MNV2PE_0_5_224_FP16))

    def test_mobilenetv2_pose_estimator_fp32(self):
        self._test_model(vg.MobileNetV2PoseEstimator.create(vg.MobileNetV2PoseEstimatorConfig.MNV2PE_0_5_224_FP32))

    def test_movenet_pose_estimator_single_fp16(self):
        self._test_model(vg.MoveNetPoseEstimator.create(vg.MoveNetConfig.MoveNet_Single_Lightning_FP16))

    def test_movenet_pose_estimator_single_fp32(self):
        self._test_model(vg.MoveNetPoseEstimator.create(vg.MoveNetConfig.MoveNet_Single_Lightning_FP32))

    def test_movenet_pose_estimator_multi_192_fp32(self):
        self._test_model(vg.MoveNetPoseEstimator.create(vg.MoveNetConfig.MoveNet_MultiPose_192x192_FP32))

    def test_movenet_pose_estimator_multi_256_fp32(self):
        self._test_model(vg.MoveNetPoseEstimator.create(vg.MoveNetConfig.MoveNet_MultiPose_256x256_FP32))

    def test_movenet_pose_estimator_multi_320_fp32(self):
        self._test_model(vg.MoveNetPoseEstimator.create(vg.MoveNetConfig.MoveNet_MultiPose_320x320_FP32))

    def test_openpose_estimator_int8(self):
        self._test_model(vg.OpenPoseEstimator.create(vg.OpenPoseConfig.LightWeightOpenPose_INT8))

    def test_openpose_estimator_fp16(self):
        self._test_model(vg.OpenPoseEstimator.create(vg.OpenPoseConfig.LightWeightOpenPose_FP16))

    def test_openpose_estimator_fp32(self):
        self._test_model(vg.OpenPoseEstimator.create(vg.OpenPoseConfig.LightWeightOpenPose_FP32))

    def test_kapao_n_coco_640_estimator(self):
        self._test_model(vg.KAPAOPoseEstimator.create(vg.KAPAOPoseConfig.KAPAO_N_COCO_640))

    def test_kapao_s_coco_640_estimator(self):
        self._test_model(vg.KAPAOPoseEstimator.create(vg.KAPAOPoseConfig.KAPAO_S_COCO_640))

    def test_kapao_s_coco_1280_estimator(self):
        self._test_model(vg.KAPAOPoseEstimator.create(vg.KAPAOPoseConfig.KAPAO_S_COCO_1280))

    def test_kapao_l_coco_1280_estimator(self):
        self._test_model(vg.KAPAOPoseEstimator.create(vg.KAPAOPoseConfig.KAPAO_L_COCO_1280))

    def test_pose_embedding(self):
        data = list(np.abs(np.random.sample((17, 4))))
        data = [tuple(d) for d in data]
        landmarks: VectorNumpy4D = list_of_vector4D(data)
        pose = COCOPose(1.0, landmarks)

        embedder = LandmarkEmbedder(PoseUtils.embed_pose)
        embedder.setup()
        _ = embedder.process(ResultList([pose]))
        embedder.release()


if __name__ == '__main__':
    unittest.main()
