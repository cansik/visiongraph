from typing import Tuple

import cv2
import numpy as np

import visiongraph as vg
from visiongraph import BoundingBox2D

from visiongraph.util import ImageUtils


def _resize_and_pad(image: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, BoundingBox2D]:
    # resize input image
    h, w = image.shape[:2]

    if w > h:
        nh = int(width / w * h)
        image = cv2.resize(image, (width, nh))
    else:
        nw = int(height / h * w)
        image = cv2.resize(image, (nw, height))

    h, w = image.shape[:2]

    # get final image size
    size = h if h > w else w

    # todo: check if width and height are different (non-square padding)

    #  create base image with background color
    background = np.zeros([size, size, 3], dtype=np.uint8)

    # add image into center
    xs = round((size - w) * 0.5)
    ys = round((size - h) * 0.5)

    background[ys:ys + h, xs:xs + w] = image

    return background, vg.BoundingBox2D(xs, ys, w, h)

iw, ih = 224, 224
image = cv2.imread("media/pose_slim.png")
h, w = image.shape[:2]

input, padding_box = ImageUtils.resize_and_pad(image, (iw, ih))
input2, padding_box2 = _resize_and_pad(image, iw, ih)
# padding_box = padding_box.scale(1.0 / iw, 1.0 / ih)

result = vg.ObjectDetectionResult(0, "face", 1.0, vg.BoundingBox2D(105 / iw, 31 / ih, 18 / iw, 20 / ih))
result.tracking_id = 2

result.annotate(input)
cv2.imshow("Input", input)

# map result
bbox = result.bounding_box
bbox.x_min = (bbox.x_min * iw - padding_box.x_min) / padding_box.width
bbox.y_min = (bbox.y_min * ih - padding_box.y_min) / padding_box.height
bbox.width = bbox.width * iw / padding_box.width
bbox.height = bbox.height * ih / padding_box.height

result.annotate(image)
cv2.imshow("Result", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
