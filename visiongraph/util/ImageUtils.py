from typing import Tuple

import cv2
import numpy as np

from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.util.MathUtils import constrain


def resize_and_letter_box(image, width, height):
    """
    Letter box (black bars) a color image (think pan & scan movie shown
    on widescreen) if not same aspect ratio as specified rows and cols.

    :param image: Input image of shape (image_rows, image_cols, channels) with dtype numpy.uint8.
    :param width: Desired columns of letter boxed image returned.
    :param height: Desired rows of letter boxed image returned.

    :return: Letter boxed image of shape (rows, cols, channels) with dtype numpy.uint8.
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


def resize_and_pad(image: np.ndarray, new_size: Tuple[int, int],
                   color: Tuple[int, int, int] = (125, 125, 125)) -> Tuple[np.ndarray, BoundingBox2D]:
    """
    Resize an image and pad it with a specified color to fit the new size.

    :param image: Input image to resize.
    :param new_size: Tuple containing the desired width and height.
    :param color: Color for padding. Defaults to (125, 125, 125).

    :return: Resize and padded image along with the bounding box of the resized image.
    """
    in_h, in_w = image.shape[:2]
    new_w, new_h = new_size
    scale = min(new_w / in_w, new_h / in_h)
    scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
    resized_img = cv2.resize(image, (scale_new_w, scale_new_h))
    d_w = max(new_w - scale_new_w, 0)
    d_h = max(new_h - scale_new_h, 0)
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    result = cv2.copyMakeBorder(resized_img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return result, BoundingBox2D(left, top, scale_new_w, scale_new_h)


def extract_roi_safe(image: np.ndarray,
                     xmin: float, ymin: float, xmax: float, ymax: float,
                     rectified: bool = False) -> Tuple[np.ndarray, int, int]:
    """
    Safely extract a region of interest (ROI) from an image based on normalized coordinates.

    :param image: Input image from which to extract the ROI.
    :param xmin: Normalized minimum x-coordinate of the ROI.
    :param ymin: Normalized minimum y-coordinate of the ROI.
    :param xmax: Normalized maximum x-coordinate of the ROI.
    :param ymax: Normalized maximum y-coordinate of the ROI.
    :param rectified: Whether to rectify the ROI. Defaults to False.

    :return: Extracted ROI and the absolute x, y coordinates in the image.
    """
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
    """
    Align an image based on an affine transformation defined by source and destination triangles.

    :param image: Input image to align.
    :param src_triangle: Source triangle points for affine transformation.
    :param dest_triangle: Destination triangle points for affine transformation.

    :return: Affine transformation matrix and the aligned image.
    """
    warp_mat = cv2.getAffineTransform(src_triangle, dest_triangle)
    warp_dst = cv2.warpAffine(image, warp_mat, (image.shape[1], image.shape[0]))
    return warp_mat, warp_dst


def roi(image: np.ndarray, box: BoundingBox2D) -> np.ndarray:
    """
    Extract ROI with absolute bounding box.

    :param image: Input image.
    :param box: Bounding box defining the ROI.

    :return: Extracted region of interest.
    """
    return image[int(box.y_min):int(box.y_min + box.height), int(box.x_min):int(box.x_min + box.width)]


def roi_safe(image: np.ndarray, box: BoundingBox2D, rectified: bool = False) -> Tuple[np.ndarray, int, int]:
    """
    Extract safe ROI with normalized bounding box.

    :param image: Input image.
    :param box: Bounding box defining the ROI.
    :param rectified: Whether to rectify the ROI. Defaults to False.

    :return: Extracted region of interest and absolute x, y coordinates.
    """
    return extract_roi_safe(image, box.x_min, box.y_min, box.x_max, box.y_max, rectified=rectified)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to an image using a bitwise AND operation.

    :param image: Input image on which to apply the mask.
    :param mask: Mask to apply, should have the same dimensions as image.

    :return: Resulting image after applying the mask.
    """
    return cv2.bitwise_and(image, image, mask=mask)
