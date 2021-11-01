from abc import abstractmethod, ABC
from typing import Optional

import numpy as np

from visiongraph.PipelineStep import PipelineStep


class BaseInput(PipelineStep, ABC):
    @abstractmethod
    def read(self) -> (int, Optional[np.ndarray]):
        pass
