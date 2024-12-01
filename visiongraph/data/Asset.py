from abc import ABC, abstractmethod


class Asset(ABC):
    """
    An abstract base class representing a digital asset.
    """

    @property
    @abstractmethod
    def exists(self) -> bool:
        """
        Determines whether the asset exists.

        :return: True if the asset exists, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def path(self) -> str:
        """
        Retrieves the file or directory path of the asset.

        :return: The file or directory path of the asset.
        """
        pass

    def prepare(self) -> bool:
        """
        Prepares the asset for use. This method may raise an exception if
        the asset is invalid or cannot be prepared.

        :return: True if the asset can be prepared, False otherwise.
        """
        pass

    @staticmethod
    def prepare_all(*assets: "Asset") -> None:
        """
        Prepares a list of assets for use. Each asset's prepare method is
        called until all assets are successfully prepared or an exception is raised.

        :param *assets: A variable number of assets to be prepared.
        """
        for asset in assets:
            asset.prepare()
