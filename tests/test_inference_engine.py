import unittest
from typing import Callable, Optional

import cv2

from visiongraph import vg


class TestInferenceEngine(unittest.TestCase):

    @staticmethod
    def _engine_test(engine_type: vg.InferenceEngine, test_method: Optional[Callable[[vg.BaseVisionEngine], None]]):
        asset = vg.KAPAOPoseConfig.KAPAO_N_COCO_640.value[0]

        engine = vg.InferenceEngineFactory.create(engine_type, [asset])
        engine.setup()
        if test_method is not None:
            test_method(engine)
        engine.release()

    def test_onnx_inference_engine_process(self):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        def method(engine: vg.BaseVisionEngine):
            engine.process(image)

        self._engine_test(vg.InferenceEngine.ONNX, method)

    def test_openvino_inference_engine_process(self):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        def method(engine: vg.BaseVisionEngine):
            engine.process(image)

        self._engine_test(vg.InferenceEngine.OpenVINO, method)

    def test_openvino2_inference_engine_process(self):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        def method(engine: vg.BaseVisionEngine):
            engine.process(image)

        self._engine_test(vg.InferenceEngine.OpenVINO2, method)

    def test_onnx_inference_engine_layers(self):
        def method(engine: vg.BaseVisionEngine):
            layers = engine.get_input_layers()
            assert layers[0].name == "images"

        self._engine_test(vg.InferenceEngine.ONNX, method)

    def test_openvino2_inference_engine_layers(self):
        def method(engine: vg.BaseVisionEngine):
            layers = engine.get_input_layers()
            assert layers[0].name == "images"

        self._engine_test(vg.InferenceEngine.OpenVINO2, method)

    def test_openvino_utils_get_inference_engine_device(self):
        device = vg.get_inference_engine_device()
        assert device


if __name__ == '__main__':
    unittest.main()
