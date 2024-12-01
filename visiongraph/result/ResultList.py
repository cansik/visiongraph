from typing import TypeVar, List, Optional, Sequence

import numpy as np

from visiongraph.result.BaseResult import BaseResult

ResultType = TypeVar('ResultType', bound=BaseResult)


class ResultList(List[ResultType], BaseResult):
    """
    A list of results that can be annotated on an image.
    """

    def __init__(self, base_list: Optional[Sequence[BaseResult]] = ()):
        """
        Initializes the ResultList object.

        :param base_list: The initial list of results. Defaults to ().

        :raises ValueError: If the input is not a sequence.
        """
        super().__init__(base_list)

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Annotates each result in the list on the given image.

        :param image: The image to annotate.
        :param **kwargs: Additional keyword arguments to be passed to the annotation method of each result.

        :raises ValueError: If any result is None.
        """
        for result in self:
            if result is None:
                raise ValueError("Cannot annotate a None result")
            result.annotate(image, **kwargs)
