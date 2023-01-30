import numpy as np
import vector

from visiongraph.model.CameraIntrinsics import CameraIntrinsics


def project_point_to_pixel(point: vector.Vector3D, intrinsics: CameraIntrinsics) -> vector.Vector2D:
    x = point.x / point.z
    y = point.y / point.z

    # todo: add distortion coefficients
    # https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L15

    u = x * intrinsics.fx + intrinsics.px
    v = y * intrinsics.fy + intrinsics.py

    return vector.obj(x=u, y=v)


def project_points_to_pixels(points: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """
    Project 3d points into 2d image space.

    Parameters:
        points: Numpy array of pixels (n, 3)
        intrinsics: Camera intrinsics
    Return:
        Numpy array of 2d positions (n, 2)
    """
    pixels = np.zeros((points.shape[0], 3), dtype=float)

    pixels[:, 0] = points[0] / points[2]
    pixels[:, 1] = points[1] / points[2]

    # todo: add distortion coefficients
    # https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L15

    pixels[:, 0] = pixels[:, 0] * intrinsics.fx + intrinsics.px
    pixels[:, 1] = pixels[:, 1] * intrinsics.fy + intrinsics.py

    return pixels


def project_pixel_to_point(pixel: vector.Vector2D, depth: float, intrinsics: CameraIntrinsics) -> vector.Vector3D:
    x = (pixel.x - intrinsics.px) / intrinsics.fx
    y = (pixel.y - intrinsics.py) / intrinsics.fy

    # todo: add distortion coefficients
    # https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L46

    return vector.obj(x=x * depth, y=y * depth, z=depth)


def project_pixels_to_points(pixels: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    """
    Project 2d pixels into 3d camera space.

    Parameters:
        pixels: Numpy array of pixels (n, 2)
        depth: Numpy array of depth values (n)
        intrinsics: Camera intrinsics
    Return:
        Numpy array of 3d positions (n, 3)
    """
    points = np.zeros((pixels.shape[0], 3), dtype=float)

    points[:, 0] = (pixels[:, 0] - intrinsics.px) / intrinsics.fx
    points[:, 1] = (pixels[:, 1] - intrinsics.py) / intrinsics.fy

    # todo: add distortion coefficients
    # https://github.com/IntelRealSense/librealsense/blob/5e73f7bb906a3cbec8ae43e888f182cc56c18692/include/librealsense2/rsutil.h#L46

    points[:, 0] *= depth
    points[:, 1] *= depth
    points[:, 2] = depth

    return points
