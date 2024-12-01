from typing import List, Tuple, Sequence, Union

import numpy as np
import vector
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from vector._methods import VectorProtocolPlanar

from visiongraph.util.CodeUtils import deprecated


def list_of_vector2D(data: List[Tuple[float, float]]) -> vector.VectorNumpy2D:
    """
    Converts a list of 2D tuples into a VectorNumpy2D.

    :param data: List of tuples, each containing two float values.

    :return: A VectorNumpy2D representation of the input data.
    """
    return vector.array(
        data, dtype=[("x", float), ("y", float)]
    ).view(vector.VectorNumpy2D)


def list_of_vector3D(data: List[Tuple[float, float, float]]) -> vector.VectorNumpy3D:
    """
    Converts a list of 3D tuples into a VectorNumpy3D.

    :param data: List of tuples, each containing three float values.

    :return: A VectorNumpy3D representation of the input data.
    """
    return vector.array(
        data, dtype=[("x", float), ("y", float), ("z", float)]
    ).view(vector.VectorNumpy3D)


def list_of_vector4D(data: List[Tuple[float, float, float, float]]) -> vector.VectorNumpy4D:
    """
    Converts a list of 4D tuples into a VectorNumpy4D.

    :param data: List of tuples, each containing four float values.

    :return: A VectorNumpy4D representation of the input data.
    """
    return vector.array(
        data, dtype=[("x", float), ("y", float), ("z", float), ("t", float)]
    ).view(vector.VectorNumpy4D)


def vector_to_array(vectors: vector.VectorNumpy) -> np.ndarray:
    """
    Converts a VectorNumpy to a NumPy ndarray.

    :param vectors: The vector to be converted.

    :return: The corresponding NumPy ndarray representation of the vectors.
    """
    return structured_to_unstructured(np.asarray(vectors))


def array_to_vector(data: np.ndarray) -> vector.VectorNumpy:
    """
    Converts a NumPy ndarray to a VectorNumpy based on its shape.

    :param data: The NumPy array to be converted.

    :return: The resulting vector representation.

    :raises Exception: If the shape of the input array is not valid for a vector.
    """
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
    """
    Converts a vector to a list of its components.

    :param v: The vector to be converted.

    :return: A list containing the vector's components.

    :raises Exception: If the vector type is unsupported.
    """
    if isinstance(v, vector.Vector2D):
        return [v.x, v.y]
    elif isinstance(v, vector.Vector3D):
        return [v.x, v.y, v.z]
    elif isinstance(v, vector.Vector4D):
        return [v.x, v.y, v.z, v.t]

    raise Exception(f"Vector {v} can not be converted to list.")


@deprecated("Please use lerp_vector_4d() instead.")
def lerp4d(a: vector.VectorNumpy4D, b: vector.VectorNumpy4D, amt: float) -> vector.VectorNumpy4D:
    """
    Linearly interpolates between two 4D vectors.

    :param a: The starting vector.
    :param b: The ending vector.
    :param amt: The interpolation amount (0.0 to 1.0).

    :return: The interpolated vector.
    """
    return lerp_vector_4d(a, b, amt)


def lerp_vector_2d(a: Union[vector.Vector2D, VectorProtocolPlanar],
                   b: Union[vector.Vector2D, VectorProtocolPlanar], amt: float) -> vector.Vector2D:
    """
    Performs linear interpolation between two 2D vectors.

    :param a: The first vector.
    :param b: The second vector.
    :param amt: The interpolation amount (0.0 to 1.0).

    :return: The resulting interpolated vector.
    """
    return vector.obj(
        x=(a.x * (1.0 - amt)) + (b.x * amt),
        y=(a.y * (1.0 - amt)) + (b.y * amt)
    )


def lerp_vector_3d(a: Union[vector.Vector3D, VectorProtocolPlanar],
                   b: Union[vector.Vector3D, VectorProtocolPlanar], amt: float) -> vector.Vector3D:
    """
    Performs linear interpolation between two 3D vectors.

    :param a: The first vector.
    :param b: The second vector.
    :param amt: The interpolation amount (0.0 to 1.0).

    :return: The resulting interpolated vector.
    """
    return vector.obj(
        x=(a.x * (1.0 - amt)) + (b.x * amt),
        y=(a.y * (1.0 - amt)) + (b.y * amt),
        z=(a.z * (1.0 - amt)) + (b.z * amt)
    )


def lerp_vector_4d(a: Union[vector.Vector4D, VectorProtocolPlanar],
                   b: Union[vector.Vector4D, VectorProtocolPlanar], amt: float) -> vector.Vector4D:
    """
    Performs linear interpolation between two 4D vectors.

    :param a: The first vector.
    :param b: The second vector.
    :param amt: The interpolation amount (0.0 to 1.0).

    :return: The resulting interpolated vector.
    """
    return vector.obj(
        x=(a.x * (1.0 - amt)) + (b.x * amt),
        y=(a.y * (1.0 - amt)) + (b.y * amt),
        z=(a.z * (1.0 - amt)) + (b.z * amt),
        t=(a.t * (1.0 - amt)) + (b.t * amt)
    )


def landmarks_center_by_indices(landmarks: vector.VectorNumpy4D, indices: Sequence[int]) -> vector.Vector4D:
    """
    Computes the center (average) of specified landmarks.

    :param landmarks: The set of landmarks.
    :param indices: The indices of the landmarks to average.

    :return: The average landmark as a vector.
    """
    x = np.average(landmarks.x[indices])
    y = np.average(landmarks.y[indices])
    z = np.average(landmarks.z[indices])
    t = np.average(landmarks.t[indices])
    return vector.obj(x=x, y=y, z=z, t=t)


def vector_distance(a: vector._methods.VectorProtocol, b: vector._methods.VectorProtocol) -> float:
    """
    Calculates the distance between two vectors.

    :param a: The first vector.
    :param b: The second vector.

    :return: The distance between the two vectors.
    """
    return abs(b.subtract(a))
