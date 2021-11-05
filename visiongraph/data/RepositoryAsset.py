import os
from typing import Optional, Tuple

from visiongraph.data.Asset import Asset
from visiongraph.util.NetworkUtils import PUBLIC_DATA_URL, prepare_data_file


class RepositoryAsset(Asset):
    def __init__(self, name: str, repository_url: str = PUBLIC_DATA_URL):
        self.name = name
        self._local_path: Optional[str] = None
        self.repository_url = repository_url

    @property
    def exists(self) -> bool:
        return self._local_path is not None and os.path.exists(self._local_path)

    @property
    def path(self) -> str:
        if self.exists:
            return self._local_path

        self.prepare()
        return os.path.abspath(self._local_path)

    def prepare(self):
        self._local_path = prepare_data_file(self.name, f"{self.repository_url}{self.name}")

    def __repr__(self):
        return self.name

    @staticmethod
    def openVino(name: str, repository_url: str = PUBLIC_DATA_URL) -> Tuple["RepositoryAsset", "RepositoryAsset"]:
        return RepositoryAsset(f"{name}.xml", repository_url), RepositoryAsset(f"{name}.bin", repository_url)
