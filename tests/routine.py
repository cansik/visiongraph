import sys
import unittest

from visiongraph import vg
from visiongraph.model._ImportStub import _ImportStub


ROUTINE_TEST_MODULES = (
    "tests.test_base_vision_engine",
    "tests.test_centroid_tracker",
    "tests.test_download_model_licenses",
    "tests.test_graph_node",
    "tests.test_lazy_import",
    "tests.test_linalg_utils",
    "tests.test_math_utils",
    "tests.test_network_utils",
    "tests.test_one_euro_filter",
    "tests.test_repo_maintenance_scripts",
    "tests.test_repository_asset_metadata",
    "tests.test_result_utils",
    "tests.test_tracker",
    "tests.test_video_capture_backend",
)


class OptionalBackendImportTests(unittest.TestCase):
    def test_public_classes_import_with_the_all_extra(self):
        classes = [
            "OakDInput",
            "RealSenseInput",
            "CamGearInput",
            "VidGearVideoRecorder",
            "MediaPipeFaceMeshEstimator",
            "ONNXVisionEngine",
            "OpenVinoEngine",
            "OneEuroFilterNumba",
            "FaissKNNClassifier",
            "MotpyTracker",
        ]
        if sys.platform in ("linux", "win32"):
            classes.append("AzureKinectInput")
        if sys.platform == "win32":
            classes.append("SpoutServer")
        if sys.platform == "darwin":
            classes.append("SyphonServer")

        for name in classes:
            with self.subTest(name=name):
                print(f"Importing vg.{name}", flush=True)
                backend = getattr(vg, name)
                if issubclass(backend, _ImportStub):
                    self.fail(backend._error_message())


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, _pattern: str | None) -> unittest.TestSuite:
    """Add the model-free test modules to the routine suite."""
    tests.addTests(loader.loadTestsFromNames(ROUTINE_TEST_MODULES))
    return tests
