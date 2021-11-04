from abc import ABC, abstractmethod

import numpy as np


class BaseResult(ABC):
    @abstractmethod
    def annotate(self, image: np.ndarray, **kwargs):
        pass
