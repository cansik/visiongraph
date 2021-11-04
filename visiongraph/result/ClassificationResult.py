import numpy as np

from visiongraph.result.BaseResult import BaseResult


class ClassificationResult(BaseResult):
    def __init__(self, class_id: int, class_name: str, score: float):
        self.class_id = class_id
        self.class_name = class_name
        self.score = score

    def annotate(self, image: np.ndarray, **kwargs):
        super().annotate(image, **kwargs)
