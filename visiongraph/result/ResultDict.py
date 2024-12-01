from typing import TypeVar, Dict

import numpy as np

from visiongraph.result.BaseResult import BaseResult

DEFAULT_IMAGE_KEY = "image"
"""
Default key for referencing images in the result dictionary.
"""

ResultType = TypeVar('ResultType', bound=BaseResult)


class ResultDict(Dict[str, ResultType], BaseResult):
    """
    A dictionary-like class that stores results indexed by string keys.

    Inherits from `BaseResult` to provide additional result-related functionality.
    """

    def __init__(self):
        """
        Initializes an empty ResultDict instance.
        """
        super().__init__()

    def annotate(self, image: np.ndarray, **kwargs):
        """
        Annotates the given image using the results stored in the dictionary.

        :param image: The image to be annotated.
        :param **kwargs: Additional keyword arguments to be passed to the annotate method of BaseResult.
        """
        for result in self.values():
            if isinstance(result, BaseResult):
                result.annotate(image, **kwargs)
