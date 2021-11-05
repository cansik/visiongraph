from typing import List, Tuple

import vector


def list_of_vector4D(data: List[Tuple[float, float, float, float]]) -> vector.VectorNumpy4D:
    return vector.array(data, dtype=[("x", float), ("y", float), ("z", float), ("t", float)]).view(vector.VectorNumpy4D)
