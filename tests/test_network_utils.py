import os
import tempfile
import unittest
from unittest.mock import patch

from visiongraph.util.NetworkUtils import ASSET_DIR_ENV_VAR, prepare_data_file


class NetworkUtilsTests(unittest.TestCase):
    def test_prepare_data_file_defaults_to_user_asset_directory(self):
        with tempfile.TemporaryDirectory() as home_dir:
            env_updates = {"HOME": home_dir}
            with patch.dict(os.environ, env_updates, clear=False):
                os.environ.pop(ASSET_DIR_ENV_VAR, None)

                def fake_download(_url, path, _description, with_progress=True, headers=None):
                    del with_progress, headers
                    with open(path, "wb") as f:
                        f.write(b"model-data")

                with patch("visiongraph.util.NetworkUtils.download_file", side_effect=fake_download):
                    file_path = prepare_data_file("default-model.onnx", "https://example.com/default-model.onnx")

                expected_dir = os.path.join(home_dir, ".visiongraph", "assets")
                self.assertEqual(os.path.join(expected_dir, "default-model.onnx"), file_path)
                self.assertTrue(os.path.exists(file_path))

    def test_prepare_data_file_allows_asset_directory_override(self):
        with tempfile.TemporaryDirectory() as asset_dir:
            with patch.dict(os.environ, {ASSET_DIR_ENV_VAR: asset_dir}, clear=False):

                def fake_download(_url, path, _description, with_progress=True, headers=None):
                    del with_progress, headers
                    with open(path, "wb") as f:
                        f.write(b"model-data")

                with patch("visiongraph.util.NetworkUtils.download_file", side_effect=fake_download):
                    file_path = prepare_data_file("override-model.onnx", "https://example.com/override-model.onnx")

                self.assertEqual(os.path.join(asset_dir, "override-model.onnx"), file_path)
                self.assertTrue(os.path.exists(file_path))


if __name__ == "__main__":
    unittest.main()
