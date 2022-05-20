from typing import Tuple

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.util.MathUtils import constrain


def resize_and_letter_box(image, width, height):
    """
    Letter box (black bars) a color image (think pan & scan movie shown
    on widescreen) if not same aspect ratio as specified rows and cols.
    :param image: numpy.ndarray((image_rows, image_cols, channels), dtype=numpy.uint8)
    :param width: int cols of letter boxed image returned
    :param height: int rows of letter boxed image returned
    :return: numpy.ndarray((rows, cols, channels), dtype=numpy.uint8)
    From: https://stackoverflow.com/a/53623469/1138326
    """
    image_rows, image_cols = image.shape[:2]
    row_ratio = height / float(image_rows)
    col_ratio = width / float(image_cols)
    ratio = min(row_ratio, col_ratio)
    image_resized = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
    letter_box = np.zeros((int(height), int(width), 3))
    row_start = int((letter_box.shape[0] - image_resized.shape[0]) / 2)
    col_start = int((letter_box.shape[1] - image_resized.shape[1]) / 2)
    letter_box[row_start:row_start + image_resized.shape[0],
    col_start:col_start + image_resized.shape[1]] = image_resized
    return letter_box


def extract_roi_safe(image: np.ndarray,
                     xmin: float, ymin: float, xmax: float, ymax: float,
                     rectified: bool = False) -> Tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]

    xs = constrain(round(xmin * w), upper=w - 1)
    ys = constrain(round(ymin * h), upper=h - 1)
    xe = constrain(round(xmax * w), upper=w - 1)
    ye = constrain(round(ymax * h), upper=h - 1)

    rw = xe - xs
    hw = ye - ys

    if not rectified:
        return image[ys:ye, xs:xe], int(xs), int(ys)

    if rw > hw:
        diff = (rw - hw) * 0.5
        ys = constrain(round(ys - diff), upper=h - 1)
        ye = constrain(round(ye + diff), upper=h - 1)
    else:
        diff = (hw - rw) * 0.5
        xs = constrain(round(xs - diff), upper=w - 1)
        xe = constrain(round(xe + diff), upper=w - 1)

    return image[ys:ye, xs:xe], int(xs), int(ys)


def align_image(image: np.ndarray,
                src_triangle: np.ndarray,
                dest_triangle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    warp_mat = cv2.getAffineTransform(src_triangle, dest_triangle)
    warp_dst = cv2.warpAffine(image, warp_mat, (image.shape[1], image.shape[0]))
    return warp_mat, warp_dst


def roi(image: np.ndarray, box: BoundingBox2D) -> np.ndarray:
    return image[int(box.y_min):int(box.y_min + box.height), int(box.x_min):int(box.x_min + box.width)]


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image, image, mask=mask)
