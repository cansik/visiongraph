from typing import Sequence, Union

import cv2
import numpy as np
import vector

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D

"""
A sequence of color tuples representing different color codes used in visualizations.
"""
COLOR_SEQUENCE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (255, 255, 255)
]

"""
A sequence of colors defined for the COCO dataset to represent different classes.
"""
COCO80_COLORS = [
    (0, 113, 188),
    (216, 82, 24),
    (236, 176, 31),
    (125, 46, 141),
    (118, 171, 47),
    (76, 189, 237),
    (161, 19, 46),
    (76, 76, 76),
    (153, 153, 153),
    (255, 0, 0),
    (255, 127, 0),
    (190, 190, 0),
    (0, 255, 0),
    (0, 0, 255),
    (170, 0, 255),
    (84, 84, 0),
    (84, 170, 0),
    (84, 255, 0),
    (170, 84, 0),
    (170, 170, 0),
    (170, 255, 0),
    (255, 84, 0),
    (255, 170, 0),
    (255, 255, 0),
    (0, 84, 127),
    (0, 170, 127),
    (0, 255, 127),
    (84, 0, 127),
    (84, 84, 127),
    (84, 170, 127),
    (84, 255, 127),
    (170, 0, 127),
    (170, 84, 127),
    (170, 170, 127),
    (170, 255, 127),
    (255, 0, 127),
    (255, 84, 127),
    (255, 170, 127),
    (255, 255, 127),
    (0, 84, 255),
    (0, 170, 255),
    (0, 255, 255),
    (84, 0, 255),
    (84, 84, 255),
    (84, 170, 255),
    (84, 255, 255),
    (170, 0, 255),
    (170, 84, 255),
    (170, 170, 255),
    (170, 255, 255),
    (255, 0, 255),
    (255, 84, 255),
    (255, 170, 255),
    (42, 0, 0),
    (84, 0, 0),
    (127, 0, 0),
    (170, 0, 0),
    (212, 0, 0),
    (255, 0, 0),
    (0, 42, 0),
    (0, 84, 0),
    (0, 127, 0),
    (0, 170, 0),
    (0, 212, 0),
    (0, 255, 0),
    (0, 0, 42),
    (0, 0, 84),
    (0, 0, 127),
    (0, 0, 170),
    (0, 0, 212),
    (0, 0, 255),
    (0, 0, 0),
    (36, 36, 36),
    (72, 72, 72),
    (109, 109, 109),
    (145, 145, 145),
    (182, 182, 182),
    (218, 218, 218),
    (255, 255, 255)
]

"""
Colors used for drawing axes: blue for x-axis, green for y-axis, and red for z-axis.
"""
AXIS_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0)
]


def draw_text(image: np.ndarray,
              text: str,
              position: Sequence[int],
              font: int = cv2.FONT_HERSHEY_SIMPLEX,
              font_scale: float = 1.0,
              color: Sequence[int] = (255, 255, 255),
              thickness: int = 1,
              **kwargs):
    """
    Draws text on an image at a specified position.

    :param image: The image on which to draw the text.
    :param text: The text string to be drawn.
    :param position: The (x, y) coordinates for the text position.
    :param font: The font type to use.
    :param font_scale: Scale factor for the font size.
    :param color: The color of the text in BGR format.
    :param thickness: Thickness of the text lines.
    :param **kwargs: Additional parameters for text rendering.
    """
    cv2.putText(image, text, position, font, font_scale, color, thickness, **kwargs)


def draw_text_normalized(image: np.ndarray,
                         text: str,
                         position: Union[vector.Vector2D, vector._methods.VectorProtocol],
                         font: int = cv2.FONT_HERSHEY_SIMPLEX,
                         font_scale: float = 1.0,
                         color: Sequence[int] = (255, 255, 255),
                         thickness: int = 1,
                         **kwargs):
    """
    Draws normalized text on an image, converting (0, 1) coordinate range to pixel coordinates.

    :param image: The image on which to draw the text.
    :param text: The text string to be drawn.
    :param position: The normalized (x, y) coordinates.
    :param font: The font type to use.
    :param font_scale: Scale factor for the font size.
    :param color: The color of the text in BGR format.
    :param thickness: Thickness of the text lines.
    :param **kwargs: Additional parameters for text rendering.
    """
    h, w = image.shape[:2]
    x = int(round(position.x * w))
    y = int(round(position.y * h))

    draw_text(image, text, (x, y), font, font_scale, color, thickness, **kwargs)


def draw_axis(image: np.ndarray, rotation: vector.Vector3D,
              center: vector.Vector2D, length: float = 0.1):
    """
    Draws 3D axes on a 2D image based on the given rotation and center.

    :param image: The image on which to draw the axes.
    :param rotation: The rotation angles (in degrees) around the x, y, and z axes.
    :param center: The center (origin) of the axes in normalized coordinates.
    :param length: The length of the axes to be drawn.
    """
    h, w = image.shape[:2]
    rays = [vector.obj(x=length, y=0, z=0),
            vector.obj(x=0, y=length, z=0),
            vector.obj(x=0, y=0, z=length)]

    for i, p in enumerate(rays):
        color = AXIS_COLORS[i]
        pp = p.rotate_nautical(np.radians(rotation.z), -np.radians(rotation.y), -np.radians(rotation.x))

        x = (pp.x + center.x) * w
        y = (-pp.y + center.y) * h

        cv2.line(image, (round(center.x * w), round(center.y * h)),
                 (round(x), round(y)), color=color, thickness=2)


def draw_bbox(image: np.ndarray, bbox: BoundingBox2D, color: Sequence[int], thickness: int = 2):
    """
    Draws a bounding box on the image.

    :param image: The image on which to draw the bounding box.
    :param bbox: The bounding box to be drawn.
    :param color: The color of the rectangle in BGR format.
    :param thickness: Thickness of the rectangle edges.
    """
    h, w = image.shape[:2]
    cv2.rectangle(image, (round(bbox.x_min * w), round(bbox.y_min * h)),
                  (round((bbox.x_min + bbox.width) * w), round((bbox.y_min + bbox.height) * h)),
                  color, thickness=thickness)


def draw_landmark(image: np.ndarray, landmark: vector.Vector4D,
                  color: Sequence[int] = (0, 0, 255),
                  size: int = 5,
                  thickness: int = 1,
                  draw_marker: bool = True,
                  marker_type: int = cv2.MARKER_CROSS):
    """
    Draws a landmark point on the image.

    :param image: The image on which to draw the landmark.
    :param landmark: The landmark position represented as (x, y, z, score).
    :param color: Color of the landmark point in BGR format.
    :param size: Size of the landmark point.
    :param thickness: Thickness of the landmark marker.
    :param draw_marker: Whether to draw a marker or a circle.
    :param marker_type: The type of marker to use if drawing a marker.
    """
    h, w = image.shape[:2]
    x = int(round(landmark.x * w))
    y = int(round(landmark.y * h))

    if draw_marker:
        cv2.drawMarker(image, (x, y), color, marker_type, size, thickness)
    else:
        cv2.circle(image, (x, y), size, color, thickness)
