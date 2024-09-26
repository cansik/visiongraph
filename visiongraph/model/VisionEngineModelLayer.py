from dataclasses import dataclass, field
from typing import Sequence, List

import numpy as np


@dataclass
class VisionEngineModelLayer:
    """
    Represents a layer in the vision engine model.
    """

    name: str
    shape: Sequence[int]
    numpy_dtype: np.dtype
    layer_names: List[str] = field(default_factory=lambda: [])
