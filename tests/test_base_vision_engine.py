import unittest

import numpy as np

from visiongraph.estimator.BaseVisionEngine import BaseVisionEngine
from visiongraph.model.VisionEngineOutput import VisionEngineOutput
from visiongraph.model.types.InputShapeOrder import InputShapeOrder


class _StubVisionEngine(BaseVisionEngine):
    def __init__(self, input_shape):
        super().__init__()
        self.input_names = ["input"]
        self._input_shape = input_shape

    def setup(self):
        pass

    def predict(self, inputs=None):
        return VisionEngineOutput()

    def get_input_shape(self, input_name: str):
        return self._input_shape

    def get_device_name(self) -> str:
        return "test"

    def release(self):
        pass

    def get_input_layers(self):
        return []

    def get_output_layers(self):
        return []


class BaseVisionEnginePreprocessingTests(unittest.TestCase):
    def test_pre_process_image_reports_padding_box_for_letterboxed_input(self):
        engine = _StubVisionEngine([1, 4, 4, 3])
        image = np.full((2, 4, 3), 10, dtype=np.uint8)

        blob, padding_box, image_size = engine.pre_process_image(
            image,
            "input",
            flip_channels=False,
            padding=True,
            transpose=False,
            order=InputShapeOrder.NWHC,
            dtype=np.uint8,
        )

        self.assertEqual((1, 4, 4, 3), blob.shape)
        self.assertEqual(0, padding_box.x_min)
        self.assertEqual(1, padding_box.y_min)
        self.assertEqual(4, padding_box.width)
        self.assertEqual(2, padding_box.height)
        self.assertEqual(4, image_size.width)
        self.assertEqual(4, image_size.height)

    def test_pre_process_image_converts_grayscale_to_three_channel_blob(self):
        engine = _StubVisionEngine([1, 3, 2, 2])
        image = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        blob, _, _ = engine.pre_process_image(
            image,
            "input",
            flip_channels=False,
            transpose=True,
            order=InputShapeOrder.NCHW,
            dtype=np.float32,
        )

        self.assertEqual((1, 3, 2, 2), blob.shape)
        np.testing.assert_allclose(blob[0, 0], image.astype(np.float32))
        np.testing.assert_allclose(blob[0, 1], image.astype(np.float32))
        np.testing.assert_allclose(blob[0, 2], image.astype(np.float32))

    def test_pre_process_image_applies_mean_scale_and_channel_flip(self):
        engine = _StubVisionEngine([1, 3, 1, 1])
        image = np.array([[[10, 20, 30]]], dtype=np.uint8)

        blob, _, _ = engine.pre_process_image(
            image,
            "input",
            flip_channels=True,
            scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            transpose=True,
            order=InputShapeOrder.NCHW,
            dtype=np.float32,
        )

        np.testing.assert_allclose(blob[:, :, :, :], np.array([[[[27.0]], [[18.0]], [[9.0]]]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
