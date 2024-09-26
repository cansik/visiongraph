"""
Defines an enumeration for the complexity of pose models.

The `PoseModelComplexity` class represents different levels of pose model complexity.
"""

from enum import Enum


class PoseModelComplexity(Enum):
    """
    Enumerates the possible complexities of pose models, from simple (Light) to complex (Heavy).
    """
    Light = 0
    Normal = 1
    Heavy = 2
