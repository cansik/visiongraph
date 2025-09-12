import unittest

import cv2

from tests.utils_for_testing import run_estimator_test
from visiongraph import vg


class SegmentationEstimationTests(unittest.TestCase):
    @staticmethod
    def _test_model(model: vg.InstanceSegmentationEstimator):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")
        run_estimator_test(model, image)

    def test_maskrcnn_segmentation_fp32(self):
        self._test_model(vg.MaskRCNNEstimator.create(vg.MaskRCNNConfig.EfficientNet_480_FP32))

    def test_maskrcnn_segmentation_fp16(self):
        self._test_model(vg.MaskRCNNEstimator.create(vg.MaskRCNNConfig.EfficientNet_480_FP16))

    def test_maskrcnn_segmentation_int8(self):
        self._test_model(vg.MaskRCNNEstimator.create(vg.MaskRCNNConfig.EfficientNet_480_INT8))

    def test_mediapipe_selfie_segmentation(self):
        self._test_model(vg.MediaPipeSelfieSegmentation.create())

    def test_yolact_segmentation(self):
        self._test_model(vg.YolcatEstimator.create(vg.YolactConfig.YolactEdge_MobileNetV2_550))

    def test_ultralytics_yolov8_segmentation_s(self):
        self._test_model(vg.YOLOv8SegmentationEstimator.create(vg.YOLOv8SegmentationConfig.YOLOv8_SEG_S))

    def test_modnet_segmentation(self):
        self._test_model(vg.ModNetEstimator.create())


if __name__ == "__main__":
    unittest.main()
