import argparse
import subprocess
import sys
import unittest
from unittest.mock import patch

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
    def test_oak_import_leaves_other_camera_sdks_unloaded(self):
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-c",
                "import sys\n"
                "from visiongraph import vg\n"
                "from visiongraph.input import add_input_step_choices\n"
                "from visiongraph.model._ImportStub import _ImportStub\n"
                "other_sdks = {'pyk4a', 'pyrealsense2', 'vidgear.gears', 'pyzed.sl'}\n"
                "assert (other_sdks | {'depthai'}).isdisjoint(sys.modules)\n"
                "assert not issubclass(vg.OakDInput, _ImportStub)\n"
                "assert other_sdks.isdisjoint(sys.modules)\n",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_resolves_provider_classes(self):
        from visiongraph.input import InputProviders, add_input_step_choices

        parser = argparse.ArgumentParser()
        with patch.dict(InputProviders, {"custom-image": vg.ImageInput}):
            add_input_step_choices(parser, add_params=False)
            self.assertIs(parser.parse_args(["--input", "custom-image"]).input, vg.ImageInput)

        self.assertIs(parser.parse_args(["--input", "oakd"]).input, vg.OakDInput)
        self.assertEqual("azure" in InputProviders, sys.platform == "win32")

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
        if sys.platform == "win32":
            classes.extend(("AzureKinectInput", "SpoutServer"))
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
