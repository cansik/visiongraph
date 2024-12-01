import os
from pathlib import Path
from typing import Union

from visiongraph.data.Asset import Asset


class LocalAsset(Asset):
    """
    A local asset implementation, representing a file-based asset.
    """

    def __init__(self, file_path: Union[str, os.PathLike]):
        """
        Initializes the LocalAsset object with a given file path.

        :param file_path: The absolute or relative path to the file.
        """
        self._file_path = str(Path(file_path))

    @property
    def exists(self) -> bool:
        """
        Checks if the local asset file exists at its specified path.

        :return: True if the file exists, False otherwise.
        """
        return os.path.exists(self._file_path)

    @property
    def path(self) -> str:
        """
        Returns the absolute path of the local asset file.

        :return: The absolute path to the file.
        """
        return self._file_path

    def __repr__(self):
        """
        Returns a string representation of the LocalAsset object, in the form of its file path.

        :return: A string containing the path to the local asset file.
        """
        return self._file_path
