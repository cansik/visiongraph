import cv2

"""
Mapping of rotation parameters to their corresponding OpenCV constants.
"""

RotationParameter = {
    "90": cv2.ROTATE_90_CLOCKWISE,
    "-90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}

"""
Mapping of flip parameters to their corresponding OpenCV constants.
"""

FlipParameter = {
    "h": cv2.FLIP_HORIZONTAL,  # Horizontal flipping
    "v": cv2.FLIP_VERTICAL,  # Vertical flipping
}
