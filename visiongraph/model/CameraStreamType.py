from enum import Enum


class CameraStreamType(Enum):
    """
    An enumeration of camera stream types, defining different types of video streams available.

    Available values:
        - `Color`: A color stream type.
        - `Depth`: A depth stream type.
        - `Infrared`: An infrared stream type.
    """

    Color = 0,
    Depth = 1,
    Infrared = 2
