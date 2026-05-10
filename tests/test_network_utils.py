import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from requests.exceptions import RequestException

from visiongraph.util.NetworkUtils import ASSET_DIR_ENV_VAR, download_file, handle_redirects, prepare_data_file


class NetworkUtilsTests(unittest.TestCase):
    def test_handle_redirects_resolves_relative_locations(self):
        redirect_response = MagicMock()
        redirect_response.status_code = 302
        redirect_response.headers = {"Location": "/api/resolve-cache/models/demo/file.onnx"}

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {}
        final_response.raise_for_status.return_value = None

        with patch("visiongraph.util.NetworkUtils.requests.head", side_effect=[redirect_response, final_response]):
            resolved_url = handle_redirects("https://huggingface.co/cansik/visiongraph/resolve/main/file.onnx")

        self.assertEqual("https://huggingface.co/api/resolve-cache/models/demo/file.onnx", resolved_url)

    def test_download_file_falls_back_to_plain_get_when_progress_head_fails(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.__exit__.return_value = None
        response.raise_for_status.return_value = None
        response.raw = tempfile.TemporaryFile()
        response.raw.write(b"model-data")
        response.raw.seek(0)

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "model.onnx")

            with (
                patch("visiongraph.util.NetworkUtils.handle_redirects", return_value="https://example.com/model.onnx"),
                patch("visiongraph.util.NetworkUtils.requests.head", side_effect=RequestException("head failed")),
                patch(
                    "visiongraph.util.NetworkUtils.requests.get",
                    return_value=response,
                ) as mocked_get,
                patch("visiongraph.util.NetworkUtils.logging.warning") as mocked_warning,
            ):
                download_file("https://example.com/model.onnx", target_path, with_progress=True)

            self.assertTrue(os.path.exists(target_path))
            with open(target_path, "rb") as file:
                self.assertEqual(b"model-data", file.read())

        mocked_get.assert_called()
        mocked_warning.assert_called_once()

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
