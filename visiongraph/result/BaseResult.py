from abc import ABC, abstractmethod

import numpy as np


class BaseResult(ABC):
    """
    An abstract base class representing a result object.
    """

    @abstractmethod
    def annotate(self, image: np.ndarray, **kwargs) -> None:
        """
        Applies annotations to the given image.

        :param image: The input image array.
        :param **kwargs: Additional keyword arguments for customizing the annotation process.
        """
        pass
