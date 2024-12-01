from abc import abstractmethod, ABC

import numpy as np


class BaseFilterNumpy(ABC):
    """
    Abstract base class for filter implementations using NumPy.

    This class defines the interface that must be implemented by concrete filters.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the filter to a given input array.

        :param x: The input array to be filtered.

        :return: The filtered output array.
        """
        pass
