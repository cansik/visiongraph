from typing import TypeVar, List, Optional, Sequence

import numpy as np

from visiongraph.result.BaseResult import BaseResult

ResultType = TypeVar('ResultType', bound=BaseResult)


class ResultList(List[ResultType], BaseResult):
    def __init__(self, base_list: Optional[Sequence[BaseResult]] = ()):
        super().__init__(base_list)

    def annotate(self, image: np.ndarray, **kwargs):
        for result in self:
            result.annotate(image, **kwargs)
