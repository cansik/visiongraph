import unittest

import cv2
import visiongraph as vg
from visiongraph.util import OSUtils


class TestInferenceEngine(unittest.TestCase):

    @staticmethod
    def _engine_test(engine_type: vg.InferenceEngine):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")
        asset = vg.KAPAOPoseConfig.KAPAO_N_COCO_640.value[0]

        engine = vg.InferenceEngineFactory.create(engine_type, [asset])
        engine.setup()
        engine.process(image)
        engine.release()

    def test_onnx_inference_engine(self):
        self._engine_test(vg.InferenceEngine.ONNX)

    def test_openvino_inference_engine(self):
        self._engine_test(vg.InferenceEngine.OpenVINO)

    def test_openvino2_inference_engine(self):
        self._engine_test(vg.InferenceEngine.OpenVINO2)


if __name__ == '__main__':
    unittest.main()
