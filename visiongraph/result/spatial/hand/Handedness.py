"""
Enumeration of hand handedness for a person.
"""

from enum import Enum


class Handedness(Enum):
    """
    Represents the handedness of a person.

    Possible values:
        NONE: No handedness specified.
        LEFT: The person is left-handed.
        RIGHT: The person is right-handed.
    """

    NONE = 0
    LEFT = 1
    RIGHT = 2
