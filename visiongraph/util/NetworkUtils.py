import logging
import os
import shutil
import sys
from typing import Any, Dict, Optional, Tuple

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

import visiongraph.cache

PUBLIC_DATA_HEADERS = {}

PUBLIC_DATA_URL = "https://huggingface.co/cansik/visiongraph/resolve/main/"


class HTTPDownloadError(Exception):
    """Raised when an HTTP error or connection error occurs during download."""
    pass


def handle_redirects(url: str, headers: Optional[Dict[str, Any]] = None) -> str:
    """
    Follow HTTP redirects manually, preserving headers, and raise on error.

    :param url: initial URL to request
    :param headers: optional headers to include
    :return: final resolved URL after redirects
    :raises HTTPDownloadError: on network error or HTTP status >= 400
    """
    while True:
        try:
            response = requests.head(url, headers=headers, allow_redirects=False)
        except RequestException as e:
            raise HTTPDownloadError(f"HEAD request failed for {url}: {e}") from e

        # redirect?
        if response.status_code in (301, 302, 303, 307, 308):
            url = response.headers.get("Location", url)
            continue

        # non-redirect: check for HTTP errors
        try:
            response.raise_for_status()
        except RequestException as e:
            raise HTTPDownloadError(f"HEAD request returned error {response.status_code} for {url}: {e}") from e

        break

    return url


def download_file(
        url: str,
        path: str,
        description: str = "download",
        with_progress: bool = True,
        headers: Optional[Dict[str, Any]] = None
):
    """
    Download a file from URL to local path, with optional progress and error handling.

    :param url: download URL
    :param path: local filesystem path
    :param description: progress bar description
    :param with_progress: whether to show a tqdm progress bar
    :param headers: optional headers for auth
    :raises HTTPDownloadError: on network error or HTTP status >= 400
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # resolve redirects
    resolved_url = handle_redirects(url, headers=headers)

    # simple download without progress
    if not with_progress:
        try:
            r = requests.get(resolved_url, headers=headers, stream=True)
            r.raise_for_status()
        except RequestException as e:
            raise HTTPDownloadError(f"GET request failed for {resolved_url}: {e}") from e

        with open(path, "wb") as out_file:
            shutil.copyfileobj(r.raw, out_file)
        return

    # with progress: first get content length
    try:
        head_req = requests.head(resolved_url, headers=headers)
        head_req.raise_for_status()
    except RequestException as e:
        raise HTTPDownloadError(f"HEAD request failed for {resolved_url}: {e}") from e

    filesize = int(head_req.headers.get("Content-Length", 0))

    chunk_size = 1024

    try:
        with requests.get(resolved_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f, tqdm(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    total=filesize,
                    file=sys.stdout,
                    desc=description
            ) as progress:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    written = f.write(chunk)
                    progress.update(written)
    except RequestException as e:
        raise HTTPDownloadError(f"Streaming download failed for {resolved_url}: {e}") from e


def prepare_openvino_model(model_name: str, url: Optional[str] = None) -> Tuple[str, str]:
    """
    Prepare OpenVINO model XML and BIN by downloading if needed.

    :param model_name: name of the model
    :param url: base URL or None to use PUBLIC_DATA_URL
    :return: (xml_path, bin_path)
    """
    model_path = prepare_data_file(f"{model_name}.xml", url)
    weights_path = prepare_data_file(f"{model_name}.bin", url)
    return model_path, weights_path


def prepare_data_file(
        file_name: str,
        url: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None
) -> str:
    """
    Ensure a data file is available locally, downloading if missing.

    :param file_name: filename to prepare
    :param url: optional base URL, defaults to PUBLIC_DATA_URL
    :param headers: optional headers for auth
    :return: absolute path to the data file
    :raises HTTPDownloadError: if download fails on retry
    :raises Exception: if file not found in repository
    """
    if url is None:
        url = f"{PUBLIC_DATA_URL}{file_name}"

    data_dir = os.path.abspath(os.path.dirname(visiongraph.cache.__file__))
    if hasattr(sys, "_MEIPASS"):
        data_dir = "./cache"

    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        return file_path

    temp_path = os.path.join(data_dir, f"{file_name}.tmp")
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # first attempt with progress
    try:
        download_file(url, temp_path, f"Downloading {file_name}", headers=headers)
    except HTTPDownloadError as e:
        logging.warning(f"Retrying download without progress: {e}")
        download_file(url, temp_path, f"Downloading {file_name}", with_progress=False, headers=headers)

    # verify download
    try:
        with open(temp_path, "rb") as f:
            header = f.read(9).decode(errors="ignore")
    except Exception:
        header = ""

    if header == "Not Found":
        raise Exception(f"Could not find file in repository: {file_name}")

    os.replace(temp_path, file_path)
    return file_path
