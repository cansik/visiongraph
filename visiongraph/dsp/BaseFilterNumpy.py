from abc import abstractmethod, ABC

import numpy as np


class BaseFilterNumpy(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass
