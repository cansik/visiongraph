import numpy as np

from visiongraph import vg


def run_estimator_test(estimator: vg.BaseEstimator, image: np.ndarray):
    with estimator:
        result = estimator.process(image)
        result.annotate(image)
