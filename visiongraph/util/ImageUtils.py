from typing import Tuple

import numpy as np

from visiongraph.util.MathUtils import constrain


def extract_roi_safe(image: np.ndarray,
                     xmin: float, ymin: float, xmax: float, ymax: float,
                     rectified: bool = False) -> Tuple[np.ndarray, float, float]:
    h, w = image.shape[:2]

    xs = constrain(round(xmin * w), upper=w - 1)
    ys = constrain(round(ymin * h), upper=h - 1)
    xe = constrain(round(xmax * w), upper=w - 1)
    ye = constrain(round(ymax * h), upper=h - 1)

    rw = xe - xs
    hw = ye - ys

    if not rectified:
        return image[ys:ye, xs:xe]

    if rw > hw:
        diff = (rw - hw) * 0.5
        ys = constrain(round(ys - diff), upper=h - 1)
        ye = constrain(round(ye + diff), upper=h - 1)
    else:
        diff = (hw - rw) * 0.5
        xs = constrain(round(xs - diff), upper=w - 1)
        xe = constrain(round(xe + diff), upper=w - 1)

    return image[ys:ye, xs:xe], xs, ys
