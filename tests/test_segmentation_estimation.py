import unittest

import cv2
import visiongraph as vg


class SegmentationEstimationTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.InstanceSegmentationEstimator):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        model.setup()
        model.process(image)
        model.release()

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


if __name__ == '__main__':
    unittest.main()
