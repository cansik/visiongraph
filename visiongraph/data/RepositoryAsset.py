import os
from typing import Optional, Tuple, Dict, Any

from visiongraph.data.Asset import Asset
from visiongraph.data.AssetMetadata import AssetMetadata
from visiongraph.util.NetworkUtils import PUBLIC_DATA_URL, prepare_data_file, PUBLIC_DATA_HEADERS


class RepositoryAsset(Asset):
    """
    Represents an asset stored in a repository.
    """

    def __init__(
        self,
        name: str,
        repository_url: str = PUBLIC_DATA_URL,
        headers: Optional[Dict[str, Any]] = PUBLIC_DATA_HEADERS,
        metadata: Optional[AssetMetadata] = None,
    ):
        """
        Initializes a RepositoryAsset object.

        :param name: The name of the asset.
        :param repository_url: The URL of the repository containing the asset. Defaults to PUBLIC_DATA_URL.
        :param headers: Optional header variable for authentication. Defaults to PUBLIC_DATA_HEADERS.
        :param metadata: Optional structured metadata describing the asset provenance.
        """
        self.name = name
        self._local_path: Optional[str] = None
        self.repository_url = repository_url
        self.headers = headers
        self._metadata = metadata

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

    @property
    def metadata(self) -> Optional[AssetMetadata]:
        """
        Returns the optional structured metadata associated with the asset.

        :return: The metadata associated with the asset, or None if not specified.
        """
        return self._metadata

    def prepare(self):
        """
        Prepares the asset by downloading its contents from the repository URL and saving it locally.
        """
        self._local_path = prepare_data_file(self.name, f"{self.repository_url}{self.name}", headers=self.headers)

    def __repr__(self):
        return self.name

    @staticmethod
    def openVino(
        name: str,
        repository_url: str = PUBLIC_DATA_URL,
        metadata: Optional[AssetMetadata] = None,
    ) -> Tuple["RepositoryAsset", "RepositoryAsset"]:
        """
        Helper method to download openVINO assets (XML and binary files).

        :param name: The name of the asset.
        :param repository_url: The URL of the repository containing the asset. Defaults to PUBLIC_DATA_URL.
        :param metadata: Optional structured metadata describing the asset provenance.

        :return: A tuple containing two RepositoryAsset objects representing the XML and binary files for the openVINO model.
        """
        return (
            RepositoryAsset(f"{name}.xml", repository_url, metadata=metadata),
            RepositoryAsset(f"{name}.bin", repository_url, metadata=metadata),
        )
