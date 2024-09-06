import unittest

import cv2
from visiongraph import vg
from visiongraph.util import OSUtils


class ObjectDetectionTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.ObjectDetector):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        model.setup()
        model.process(image)
        model.release()

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

    def test_ultralytics_yolov8_detector_obb_s(self):
        self._test_model(vg.YOLOv8OBBDetector.create(vg.YOLOv8OBBConfig.YOLOv8_OBB_S))

    def test_crowdhuman_detector(self):
        self._test_model(vg.CrowdHumanDetector.create(vg.CrowdHumanConfig.YOLOv5_N_640))


if __name__ == '__main__':
    unittest.main()
