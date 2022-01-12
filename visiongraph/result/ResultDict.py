from typing import TypeVar, List, Optional, Sequence, Dict

import numpy as np

from visiongraph.result.BaseResult import BaseResult

ResultType = TypeVar('ResultType', bound=BaseResult)


class ResultDict(Dict[str, ResultType], BaseResult):
    def __init__(self):
        super().__init__()

    def annotate(self, image: np.ndarray, **kwargs):
        for result in self.values():
            result.annotate(image, **kwargs)
