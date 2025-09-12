import unittest

import cv2

from tests.utils_for_testing import save_annotation_image
from visiongraph import vg


class FaceRecognitionEstimationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = vg.SpatialCascadeEstimator(
            vg.AdasFaceDetector.create(), landmarks=vg.RegressionLandmarkEstimator()
        )
        self.network.setup()

    def doCleanups(self) -> None:
        self.network.release()

    def _test_model(self, model: vg.FaceRecognitionEstimator):
        image = cv2.imread("assets/head-pexels-ike-louie-natividad-2709388.jpg")
        result = self.network.process(image)[0]

        model.setup()
        result = model.process_detection(image, result)
        result.annotate(image)
        save_annotation_image(image, self._testMethodName)
        model.release()

    def test_face_reidentification_estimator_int8(self):
        self._test_model(vg.FaceReidentificationEstimator.create(vg.FaceReidentificationConfig.Retail_0095_FP16_INT8))

    def test_face_reidentification_estimator_fp16(self):
        self._test_model(vg.FaceReidentificationEstimator.create(vg.FaceReidentificationConfig.Retail_0095_FP16))

    def test_face_reidentification_estimator_fp32(self):
        self._test_model(vg.FaceReidentificationEstimator.create(vg.FaceReidentificationConfig.Retail_0095_FP32))


if __name__ == "__main__":
    unittest.main()
