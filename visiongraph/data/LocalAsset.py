import os

from visiongraph.data.Asset import Asset


class LocalAsset(Asset):
    def __init__(self, file_path: str):
        self._file_path = file_path

    @property
    def exists(self) -> bool:
        return os.path.exists(self._file_path)

    @property
    def path(self) -> str:
        return self._file_path

    def __repr__(self):
        return self._file_path
