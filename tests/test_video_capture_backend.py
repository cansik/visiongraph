import unittest
from types import SimpleNamespace

from visiongraph.model.types.VideoCaptureBackend import _available_video_capture_backends


class VideoCaptureBackendTests(unittest.TestCase):
    def test_only_includes_constants_exposed_by_opencv_build(self):
        cv2_module = SimpleNamespace(CAP_ANY=0, CAP_V4L2=200)

        backends = _available_video_capture_backends(cv2_module)

        self.assertEqual(backends, {"any": 0, "v4l2": 200})


if __name__ == "__main__":
    unittest.main()
