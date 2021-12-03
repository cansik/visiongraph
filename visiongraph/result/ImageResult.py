import numpy as np

from visiongraph.result.BaseResult import BaseResult


class ImageResult(BaseResult):
    def __init__(self, output: np.ndarray):
        self.output = output

    def annotate(self, image: np.ndarray, **kwargs):
        super().annotate(image, **kwargs)

        # todo: implement drawing image onto other image
