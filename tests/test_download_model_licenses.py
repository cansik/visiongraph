import unittest
from unittest.mock import Mock, patch

from scripts.download_model_licenses import (
    build_output_filename,
    download_licenses,
    resolve_download_url,
)


class DownloadModelLicensesTests(unittest.TestCase):
    def test_resolve_download_url_converts_github_blob_url_to_raw_url(self):
        url = "https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE"

        resolved = resolve_download_url(url)

        self.assertEqual(
            "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/LICENSE",
            resolved,
        )

    def test_build_output_filename_is_deterministic_and_uses_text_suffix_when_missing(self):
        filename = build_output_filename(
            source_name="OpenVINO Open Model Zoo",
            license_name="Apache-2.0",
            license_url="https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE",
        )

        self.assertEqual("openvino-open-model-zoo-apache-2-0-7c7077d9.txt", filename)

    @patch("scripts.download_model_licenses.requests.get")
    def test_download_licenses_deduplicates_by_metadata_group(self, mock_get: Mock):
        response = Mock()
        response.text = "license text"
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        with patch("scripts.download_model_licenses.build_download_plan") as mock_plan:
            mock_plan.return_value = (
                {
                    (
                        "OpenVINO Open Model Zoo",
                        "https://github.com/openvinotoolkit/open_model_zoo",
                        "Apache-2.0",
                        "https://github.com/openvinotoolkit/open_model_zoo/blob/master/LICENSE",
                    ): [],
                    (
                        "Ultralytics",
                        "https://github.com/ultralytics/ultralytics",
                        "AGPL-3.0",
                        "https://github.com/ultralytics/ultralytics/blob/master/LICENSE",
                    ): [],
                },
                [],
            )

            with patch("pathlib.Path.write_text") as mock_write_text:
                written_files = download_licenses(output_dir=self._output_dir())

        self.assertEqual(2, len(written_files))
        self.assertEqual(2, mock_get.call_count)
        self.assertEqual(
            "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/LICENSE",
            mock_get.call_args_list[0].args[0],
        )
        self.assertEqual(2, mock_write_text.call_count)

    def _output_dir(self):
        from pathlib import Path

        return Path("build/licenses-test")


if __name__ == "__main__":
    unittest.main()
