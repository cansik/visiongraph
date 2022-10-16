from typing import TypeVar, Dict

import numpy as np

from visiongraph.result.BaseResult import BaseResult

DEFAULT_IMAGE_KEY = "image"

ResultType = TypeVar('ResultType', bound=BaseResult)


class ResultDict(Dict[str, ResultType], BaseResult):
    def __init__(self):
        super().__init__()

    def annotate(self, image: np.ndarray, **kwargs):
        for result in self.values():
            if isinstance(result, BaseResult):
                result.annotate(image, **kwargs)
