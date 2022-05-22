import logging
import os
import sys
from typing import Tuple

import requests
from tqdm import tqdm

import visiongraph.cache

PUBLIC_DATA_URL = "https://github.com/cansik/data-storage/releases/download/sarmotion/"


def download_file(url: str, path: str, description: str = "download"):
    head_request = requests.head(url)

    if "Content-Length" in head_request.headers:
        filesize = int(head_request.headers["Content-Length"])
    else:
        filesize = 0

    os.makedirs(os.path.dirname(path), exist_ok=True)

    dl_path = path
    chunk_size = 1024

    with requests.get(url, stream=True) as r, open(dl_path, "wb") as f, tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=filesize,
            file=sys.stdout,
            desc=description
    ) as progress:
        for chunk in r.iter_content(chunk_size=chunk_size):
            datasize = f.write(chunk)
            progress.update(datasize)


def prepare_openvino_model(model_name, url: str = None) -> Tuple[str, str]:
    model_path = prepare_data_file(f"{model_name}.xml", url)
    weights_path = prepare_data_file(f"{model_name}.bin", url)
    return model_path, weights_path


def prepare_data_file(file_name: str, url: str = None) -> str:
    if url is None:
        url = f"{PUBLIC_DATA_URL}{file_name}"

    data_path = os.path.abspath(os.path.dirname(visiongraph.cache.__file__))
    file_path = os.path.join(data_path, file_name)

    if os.path.exists(file_path):
        return file_path

    temp_file = os.path.join(data_path, f"{file_name}.tmp")

    if os.path.exists(temp_file):
        os.remove(temp_file)

    download_file(url, temp_file, f"Downloading {file_name}")

    # check if file has been downloaded correctly
    head = ""
    try:
        with open(temp_file, 'rb') as f:
            head = f.read(9).decode()
    except Exception as ex:
        logging.debug(ex)

    if head == "Not Found":
        raise Exception(f"Could not find file in repository: {file_name}")

    os.rename(temp_file, file_path)
    return file_path
