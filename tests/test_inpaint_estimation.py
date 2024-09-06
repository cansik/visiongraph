import unittest

import cv2
from visiongraph import vg


class InpaintTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.BaseInpainter):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")
        mask = cv2.imread("assets/inpaint-mask.png")

        model.setup()
        model.inpaint(image, mask)
        model.release()

    def test_gmcnn_inpainter_fp32(self):
        self._test_model(vg.GMCNNInpainter.create(vg.GMCNNConfig.GMCNN_Places2_FP32))

    def test_gmcnn_inpainter_fp16(self):
        self._test_model(vg.GMCNNInpainter.create(vg.GMCNNConfig.GMCNN_Places2_FP16))


if __name__ == '__main__':
    unittest.main()
