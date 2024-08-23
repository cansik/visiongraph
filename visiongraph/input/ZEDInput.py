import logging
from typing import Optional

import cv2
import numpy as np
import pyzed.sl as sl

from visiongraph.input.BaseDepthCamera import BaseDepthCamera
from visiongraph.model.CameraStreamType import CameraStreamType


class ZEDCapture:
    timestamp: float = 0
    left_image: sl.Mat = sl.Mat()
    depth: sl.Mat = sl.Mat()


class ZEDInput(BaseDepthCamera):
    def __init__(self):
        super().__init__()

        self.camera = sl.Camera()

        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.AUTO
        self.init_params.camera_fps = 30

        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode DEPTH_MODE.PERFORMANCE
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)

        self.capture = ZEDCapture()

        self.runtime_parameters = sl.RuntimeParameters()

        self._last_color_frame: Optional[np.ndarray] = None

    def setup(self):
        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise Exception(f"Could not start ZED camera: {err}")
        print("camera has been opened")

    def read(self) -> (int, Optional[np.ndarray]):
        err = self.camera.grab(self.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            return 0, None

        # fill capture
        self.capture.timestamp = self.camera.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_milliseconds()
        self.camera.retrieve_image(self.capture.left_image, sl.VIEW.LEFT)
        self.camera.retrieve_measure(self.capture.depth, sl.MEASURE.DEPTH)

        frame = cv2.cvtColor(self.capture.left_image.get_data(), cv2.COLOR_BGRA2BGR)
        self._last_color_frame = frame

        return 0, frame

    def release(self):
        self.camera.close()

    def distance(self, x: float, y: float) -> float:
        if not self.camera.is_opened():
            return -1

        depth_frame = self.capture.depth
        ix, iy = self._calculate_depth_coordinates(x, y, depth_frame.get_width(), depth_frame.get_height())

        err, depth_value = depth_frame.get_value(ix, iy)

        if err != sl.ERROR_CODE.SUCCESS:
            logging.warning(f"Could not read depth from ZED depth frame: {err}")
            return -1

        return depth_value / 1000

    @property
    def depth_buffer(self) -> np.ndarray:
        return self.capture.depth.get_data()

    @property
    def depth_map(self) -> np.ndarray:
        return self._colorize(self.depth_buffer, (0, 12000), cv2.COLORMAP_JET)

    @property
    def gain(self) -> int:
        return self.camera.get_camera_settings(sl.VIDEO_SETTINGS.GAIN)

    @gain.setter
    def gain(self, value: int):
        self.camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(value))

    @property
    def exposure(self) -> int:
        return self.camera.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE)

    @exposure.setter
    def exposure(self, value: int):
        pass

    @property
    def enable_auto_exposure(self) -> bool:
        pass

    @enable_auto_exposure.setter
    def enable_auto_exposure(self, value: bool):
        pass

    @property
    def enable_auto_white_balance(self) -> bool:
        pass

    @enable_auto_white_balance.setter
    def enable_auto_white_balance(self, value: bool):
        pass

    @property
    def white_balance(self) -> int:
        pass

    @white_balance.setter
    def white_balance(self, value: int):
        pass

    def get_camera_matrix(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        cam = self._get_camera_params()
        return np.array([
            [cam.fx, 0, cam.cx],
            [0, cam.fy, cam.cy],
            [0, 0, 1]
        ], dtype=float)

    def get_fisheye_distortion(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        cam = self._get_camera_params()
        return cam.disto

    def _get_camera_params(self) -> sl.CameraParameters:
        info: sl.CameraInformation = self.camera.get_camera_information()
        config: sl.CameraConfiguration = info.camera_configuration
        params: sl.CalibrationParameters = config.calibration_parameters

        left_cam: sl.CameraParameters = params.left_cam
        return left_cam

    @property
    def serial(self) -> str:
        if not self.camera.is_opened():
            return "none"

        return str(self.camera.get_camera_information().serial_number)

    def pre_process_image(self, image: np.ndarray,
                          stream_type: CameraStreamType = CameraStreamType.Color) -> Optional[np.ndarray]:
        if stream_type == CameraStreamType.Depth:
            return self._colorize(image, (0, 12000), cv2.COLORMAP_JET)

        return image

    def get_raw_image(self, stream_type: CameraStreamType = CameraStreamType.Color) -> Optional[np.ndarray]:
        if stream_type == CameraStreamType.Depth:
            return self.depth_buffer
        elif stream_type == CameraStreamType.Color:
            return self._last_color_frame

        return None
