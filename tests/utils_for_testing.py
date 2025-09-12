import os
from pathlib import Path

import cv2
import numpy as np

from visiongraph import vg


def run_estimator_test(estimator: vg.BaseEstimator, image: np.ndarray, test_method_name: str):
    with estimator:
        result = estimator.process(image)
        result.annotate(image)
        save_annotation_image(image, test_method_name)


def save_annotation_image(image: np.ndarray, name: str):
    if "SAVE_ANNOTATIONS" not in os.environ:
        return

    output = Path("annotations") / f"{name}.png"
    output.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(output), image)
