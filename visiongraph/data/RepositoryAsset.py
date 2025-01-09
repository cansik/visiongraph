import os
from typing import Optional, Tuple, Dict, Any

from visiongraph.data.Asset import Asset
from visiongraph.util.NetworkUtils import PUBLIC_DATA_URL, prepare_data_file, PUBLIC_DATA_HEADERS


class RepositoryAsset(Asset):
    """
    Represents an asset stored in a repository.
    """

    def __init__(self, name: str,
                 repository_url: str = PUBLIC_DATA_URL,
                 headers: Optional[Dict[str, Any]] = PUBLIC_DATA_HEADERS):
        """
        Initializes a RepositoryAsset object.

        :param name: The name of the asset.
        :param repository_url: The URL of the repository containing the asset. Defaults to PUBLIC_DATA_URL.
        :param headers: Optional header variable for authentication. Defaults to PUBLIC_DATA_HEADERS.
        """
        self.name = name
        self._local_path: Optional[str] = None
        self.repository_url = repository_url
        self.headers = headers

    @property
    def exists(self) -> bool:
        """
        Checks if a local copy of the asset exists.

        :return: True if a local copy exists, False otherwise.
        """
        return self._local_path is not None and os.path.exists(self._local_path)

    @property
    def path(self) -> str:
        """
        Returns the absolute path to the asset's file if it exists locally.

        If the asset does not exist locally, prepares it by downloading from the repository URL.

        :return: The local or prepared path to the asset.
        """
        if self.exists:
            return self._local_path

        self.prepare()
        return os.path.abspath(self._local_path)

    def prepare(self):
        """
        Prepares the asset by downloading its contents from the repository URL and saving it locally.
        """
        self._local_path = prepare_data_file(self.name,
                                             f"{self.repository_url}{self.name}",
                                             headers=self.headers)

    def __repr__(self):
        return self.name

    @staticmethod
    def openVino(name: str, repository_url: str = PUBLIC_DATA_URL) -> Tuple["RepositoryAsset", "RepositoryAsset"]:
        """
        Helper method to download openVINO assets (XML and binary files).

        :param name: The name of the asset.
        :param repository_url: The URL of the repository containing the asset. Defaults to PUBLIC_DATA_URL.

        :return: A tuple containing two RepositoryAsset objects representing the XML and binary files for the openVINO model.
        """
        return RepositoryAsset(f"{name}.xml", repository_url), RepositoryAsset(f"{name}.bin", repository_url)
