from enum import Enum


class RealSenseColorScheme(Enum):
    """
    Color scheme used by RealSense cameras.

    This enumeration defines the different color schemes available for RealSense cameras,
    including common ones like Jet, Classic, and Quantized, as well as some specialized ones
    like Bio and Cold. These schemes can be used to adjust the color balance of the camera's output.
    """
    Jet = 0
    Classic = 1
    WhiteToBlack = 2
    BlackToWhite = 3
    Bio = 4
    Cold = 5
    Warm = 6
    Quantized = 7
    Pattern = 8
