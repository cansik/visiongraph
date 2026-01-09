import unittest

import cv2

from tests.utils_for_testing import run_estimator_test
from visiongraph import vg
from visiongraph.util import OSUtils


class ObjectDetectionTests(unittest.TestCase):
    def _test_model(self, model: vg.ObjectDetector):
        image = cv2.imread("assets/pexels-jimbear-2926723.jpg")
        run_estimator_test(model, image, self._testMethodName)

    def test_center_net_fp16(self):
        self._test_model(vg.CenterNetDetector.create(vg.CenterNetConfig.CenterNet_FP16))

    def test_center_net_fp32(self):
        self._test_model(vg.CenterNetDetector.create(vg.CenterNetConfig.CenterNet_FP32))

    def test_detr_detector_fp16(self):
        self._test_model(vg.DETRDetector.create(vg.DETRConfig.DETR_Resnet50_FP16))

    def test_detr_detector_fp32(self):
        self._test_model(vg.DETRDetector.create(vg.DETRConfig.DETR_Resnet50_FP32))

    def test_ssd_detector_int8(self):
        self._test_model(vg.SSDDetector.create(vg.SSDConfig.PersonDetection_0200_256x256_FP16_INT8))

    def test_ssd_detector_fp16(self):
        self._test_model(vg.SSDDetector.create(vg.SSDConfig.PersonDetection_0200_256x256_FP16))

    def test_ssd_detector_fp32(self):
        self._test_model(vg.SSDDetector.create(vg.SSDConfig.PersonDetection_0200_256x256_FP32))

    def test_yolov3_detector_fp16(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOv3_FP16))

    def test_yolov3_detector_fp32(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOv3_FP32))

    def test_yolov3_tiny_detector_fp16(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOv3_Tiny_FP16))

    def test_yolov4_detector_fp16(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOv4_FP16))

    def test_yolov4_detector_fp32(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOv4_FP32))

    @unittest.skipUnless(not OSUtils.isMacOSX(), "Not supported on MacOS")
    def test_yolov4_tiny_detector_fp16(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOv4_Tiny_FP16))

    @unittest.skipUnless(not OSUtils.isMacOSX(), "Not supported on MacOS")
    def test_yolov4_tiny_detector_fp32(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOv4_Tiny_FP32))

    def test_yolovx_tiny_detector_fp16(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOX_Tiny_FP16))

    def test_yolovx_tiny_detector_fp32(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOX_Tiny_FP32))

    def test_yolovf_detector_fp16(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOF_FP16))

    def test_yolovf_detector_fp32(self):
        self._test_model(vg.YOLODetector.create(vg.YOLOConfig.YOLOF_FP32))

    def test_yolov5_detector_n(self):
        self._test_model(vg.YOLOv5Detector.create(vg.YOLOv5Config.YOLOv5_N))

    def test_yolov5_detector_s(self):
        self._test_model(vg.YOLOv5Detector.create(vg.YOLOv5Config.YOLOv5_S))

    def test_ultralytics_yolov8_detector_s(self):
        self._test_model(vg.YOLOv8Detector.create(vg.YOLOv8Config.YOLOv8_S))

    def test_ultralytics_yolov8_oiv7_detector_s(self):
        self._test_model(vg.YOLOv8Detector.create(vg.YOLOv8Config.YOLOv8_S_Open_Images_V7))

    def test_ultralytics_yolov8_detector_obb_s(self):
        self._test_model(vg.YOLOv8OBBDetector.create(vg.YOLOv8OBBConfig.YOLOv8_OBB_S))

    def test_crowdhuman_detector(self):
        self._test_model(vg.CrowdHumanDetector.create(vg.CrowdHumanConfig.YOLOv5_N_640))

    def test_deimv2_pico_detector(self):
        self._test_model(vg.DEIMv2Detector.create(vg.DEIMv2Config.DEIMv2_HgNetv2_Pico_COCO))

    def test_deimv2_s_detector(self):
        self._test_model(vg.DEIMv2Detector.create(vg.DEIMv2Config.DEIMv2_Dino3_S_COCO))

    def test_sliding_window_estimator(self):
        nms_options = vg.NMSOptions(score_threshold=0.05)
        model = vg.YOLOv8Detector.create(vg.YOLOv8Config.YOLOv8_L)
        model.nms_options.enabled = False
        model.min_score = 0.05
        self._test_model(vg.SlidingWindowEstimator(model, 128, (640, 480), min_score=0.05, nms_options=nms_options))


if __name__ == "__main__":
    unittest.main()
