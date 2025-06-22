import unittest

import cv2
import pytest

from visiongraph import vg
from visiongraph.util import OSUtils


class ImageTranslationTests(unittest.TestCase):

    @staticmethod
    def _test_model(model: vg.VisionEstimator[vg.ImageResult]):
        image = cv2.imread("assets/multi-pose-pexels-rodnae-productions-7502572.jpg")

        model.setup()
        model.process(image)
        model.release()

    def test_deblurv2_gan_fp32(self):
        self._test_model(vg.DeblurGANv2.create(vg.DeblurGANv2Config.DeblurGANv2_FP32))

    def test_deblurv2_gan_fp16(self):
        self._test_model(vg.DeblurGANv2.create(vg.DeblurGANv2Config.DeblurGANv2_FP16))

    def test_midas_depth_estimation(self):
        self._test_model(vg.MidasDepthEstimator.create(vg.MidasConfig.MidasSmall))

    def test_mbllen(self):
        self._test_model(vg.MBLLENEstimator.create())


if __name__ == '__main__':
    unittest.main()
