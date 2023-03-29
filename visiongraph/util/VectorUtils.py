from typing import List, Tuple, Sequence

import numpy as np
import vector
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

from visiongraph.util.CodeUtils import deprecated


def list_of_vector2D(data: List[Tuple[float, float]]) -> vector.VectorNumpy2D:
    return vector.array(
        data, dtype=[("x", float), ("y", float)]
    ).view(vector.VectorNumpy2D)


def list_of_vector3D(data: List[Tuple[float, float, float]]) -> vector.VectorNumpy3D:
    return vector.array(
        data, dtype=[("x", float), ("y", float), ("z", float)]
    ).view(vector.VectorNumpy3D)


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


def vector_as_list(v: vector.Vector) -> List[float]:
    if isinstance(v, vector.Vector2D):
        return [v.x, v.y]
    elif isinstance(v, vector.Vector3D):
        return [v.x, v.y, v.z]
    elif isinstance(v, vector.Vector4D):
        return [v.x, v.y, v.z, v.t]

    raise Exception(f"Vector {v} can not be converted to list.")


@deprecated("Please use lerp_vector_4d() instead.")
def lerp4d(a: vector.VectorNumpy4D, b: vector.VectorNumpy4D, amt: float) -> vector.VectorNumpy4D:
    return lerp_vector_4d(a, b, amt)


def lerp_vector_2d(a: vector.Vector2D, b: vector.Vector2D, amt: float) -> vector.Vector2D:
    return vector.obj(
        x=(a.x * (1.0 - amt)) + (b.x * amt),
        y=(a.y * (1.0 - amt)) + (b.y * amt)
    )


def lerp_vector_3d(a: vector.Vector3D, b: vector.Vector3D, amt: float) -> vector.Vector3D:
    return vector.obj(
        x=(a.x * (1.0 - amt)) + (b.x * amt),
        y=(a.y * (1.0 - amt)) + (b.y * amt),
        z=(a.z * (1.0 - amt)) + (b.z * amt)
    )


def lerp_vector_4d(a: vector.Vector4D, b: vector.Vector4D, amt: float) -> vector.Vector4D:
    return vector.obj(
        x=(a.x * (1.0 - amt)) + (b.x * amt),
        y=(a.y * (1.0 - amt)) + (b.y * amt),
        z=(a.z * (1.0 - amt)) + (b.z * amt),
        t=(a.t * (1.0 - amt)) + (b.t * amt)
    )


def landmarks_center_by_indices(landmarks: vector.VectorNumpy4D, indices: Sequence[int]) -> vector.Vector4D:
    x = np.average(landmarks.x[indices])
    y = np.average(landmarks.y[indices])
    z = np.average(landmarks.z[indices])
    t = np.average(landmarks.t[indices])
    return vector.obj(x=x, y=y, z=z, t=t)
