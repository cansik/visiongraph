from typing import List, Tuple

import numpy as np
import vector
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured


def list_of_vector4D(data: List[Tuple[float, float, float, float]]) -> vector.VectorNumpy4D:
    return vector.array(
        data, dtype=[("x", float), ("y", float), ("z", float), ("t", float)]
    ).view(vector.VectorNumpy4D)


def vector_to_array(vectors: vector.VectorNumpy) -> np.ndarray:
    return structured_to_unstructured(np.asarray(vectors))


def array_to_vector(data: np.ndarray) -> vector.VectorNumpy:
    h, w = data.shape[:2]

    data = unstructured_to_structured(data)

    if w == 2:
        return vector.array(data, dtype=[("x", float), ("y", float)]).view(vector.VectorNumpy2D)
    elif w == 3:
        return vector.array(data, dtype=[("x", float), ("y", float), ("z", float)]).view(vector.VectorNumpy3D)
    elif w == 4:
        return vector.array(data, dtype=[("x", float), ("y", float),
                                         ("z", float), ("t", float)]).view(vector.VectorNumpy4D)
    else:
        raise Exception(f"Shape ({h}, {w}) is not a valid vector numpy shape.")
