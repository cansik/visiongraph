import unittest

import cv2
from visiongraph import vg
from visiongraph.util import OSUtils


class FaceDetectionTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.FaceDetector):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        model.setup()
        model.process(image)
        model.release()

    def test_adas_face_detection_fp32(self):
        self._test_model(vg.AdasFaceDetector.create(vg.AdasFaceConfig.MobileNet_672x384_FP32))

    def test_openvino_face_detection_256_fp32(self):
        self._test_model(vg.OpenVinoFaceDetector.create(vg.OpenVinoFaceConfig.MobileNetV2_256_FP32))

    def test_openvino_face_detection_256_fp16(self):
        self._test_model(vg.OpenVinoFaceDetector.create(vg.OpenVinoFaceConfig.MobileNetV2_256_FP16))

    def test_openvino_face_detection_256_int8(self):
        self._test_model(vg.OpenVinoFaceDetector.create(vg.OpenVinoFaceConfig.MobileNetV2_256_FP16_INT8))

    def test_openvino_face_detection_640_fp32(self):
        self._test_model(vg.OpenVinoFaceDetector.create(vg.OpenVinoFaceConfig.MobileNetV2_640_FP32))

    def test_openvino_face_detection_640_fp16(self):
        self._test_model(vg.OpenVinoFaceDetector.create(vg.OpenVinoFaceConfig.MobileNetV2_640_FP16))

    def test_openvino_face_detection_640_int8(self):
        self._test_model(vg.OpenVinoFaceDetector.create(vg.OpenVinoFaceConfig.MobileNetV2_640_FP16_INT8))


if __name__ == '__main__':
    unittest.main()
