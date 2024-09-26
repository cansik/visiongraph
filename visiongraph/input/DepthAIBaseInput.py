import typing
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Optional, Tuple

import depthai as dai
import numpy as np
from depthai import CameraFeatures

from visiongraph.input.BaseCamera import BaseCamera
from visiongraph.model.CameraStreamType import CameraStreamType

_CameraProperties = dai.ColorCameraProperties


class DepthAIBaseInput(BaseCamera, ABC):
    """
    Abstract base class for DepthAI camera input handling.

    This class provides basic functionalities to manage camera properties, settings, 
    and data streams for DepthAI-compatible cameras.
    """

    def __init__(self):
        """
        Initializes the DepthAIBaseInput object with default settings for camera properties.
        """
        super().__init__()

        # settings
        self.queue_max_size: int = 1

        self.color_sensor_resolution: dai.ColorCameraProperties.SensorResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P

        self.interleaved: bool = False
        self.color_isp_scale: Optional[Tuple[int, int]] = None
        self.color_board_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A
        self.color_fps: Optional[float] = None

        self._focus_mode: dai.RawCameraControl.AutoFocusMode = dai.RawCameraControl.AutoFocusMode.AUTO
        self._manual_lens_pos: int = 0

        self._auto_exposure: bool = True
        self._exposure: timedelta = timedelta(microseconds=30)
        self._iso_sensitivity: int = 400

        self._auto_white_balance: bool = True
        self._white_balance: int = 1000

        # pipeline objects
        self.pipeline: Optional[dai.Pipeline] = None
        self.color_camera: Optional[dai.node.ColorCamera] = None
        self.device: Optional[dai.Device] = None

        # node names
        self.rgb_stream_name = "rgb"
        self.rgb_isp_stream_name = "rgb_isp"
        self.rgb_control_in_name = "rbg_control_in"

        # nodes
        self.color_x_out: Optional[dai.node.XLinkOut] = None
        self.color_isp_out: Optional[dai.node.XLinkOut] = None
        self.color_control_in: Optional[dai.node.XLinkIn] = None

        self.rgb_control_queue: Optional[dai.DataInputQueue] = None
        self.rgb_queue: Optional[dai.DataOutputQueue] = None
        self.rgb_isp_queue: Optional[dai.DataOutputQueue] = None

        # capture
        self._last_ts: int = 0
        self._last_rgb_frame: Optional[np.ndarray] = None

    def setup(self):
        """
        Sets up the camera pipeline and prepares the camera for streaming.

        This method initializes the pipeline, starts the device, and prepares the output queues for RGB streams.
        """
        self.pipeline = dai.Pipeline()
        self.pre_start_setup()

        # starts pipeline
        self.device = dai.Device(self.pipeline).__enter__()

        self.rgb_control_queue = self.device.getInputQueue(self.rgb_control_in_name)
        self.rgb_queue = self.device.getOutputQueue(name=self.rgb_stream_name, maxSize=self.queue_max_size,
                                                    blocking=False)
        self.rgb_isp_queue = self.device.getOutputQueue(name=self.rgb_isp_stream_name, maxSize=self.queue_max_size,
                                                        blocking=False)

        # wait for the first isp frame
        rgb_isp_frame = typing.cast(dai.ImgFrame, self.rgb_isp_queue.get())
        self.width = rgb_isp_frame.getWidth()
        self.height = rgb_isp_frame.getHeight()

    def pre_start_setup(self):
        """
        Prepares the camera node and sets its properties before starting the streaming process.

        This method configures the camera parameters such as resolution, interleaved mode, and stream linking.
        """
        self.color_camera = self.pipeline.create(dai.node.ColorCamera)
        self.color_camera.setBoardSocket(self.color_board_socket)
        self.color_camera.setResolution(self.color_sensor_resolution)

        if self.color_fps is not None:
            self.color_camera.setFps(self.color_fps)

        self.color_camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.color_camera.setInterleaved(self.interleaved)

        if self.color_isp_scale is not None:
            self.color_camera.setIspScale(self.color_isp_scale[0], self.color_isp_scale[1])

        self.color_x_out = self.pipeline.create(dai.node.XLinkOut)
        self.color_x_out.setStreamName(self.rgb_stream_name)
        self.color_camera.video.link(self.color_x_out.input)

        self.color_isp_out = self.pipeline.create(dai.node.XLinkOut)
        self.color_isp_out.setStreamName(self.rgb_isp_stream_name)
        self.color_camera.isp.link(self.color_isp_out.input)

        self.color_control_in = self.pipeline.create(dai.node.XLinkIn)
        self.color_control_in.setStreamName(self.rgb_control_in_name)
        self.color_control_in.out.link(self.color_camera.inputControl)

    @abstractmethod
    def read(self) -> (int, Optional[np.ndarray]):
        """
        Reads the next RGB frame from the camera queue and updates internal properties.

        Returns:
            Tuple[int, Optional[np.ndarray]]: The timestamp of the frame and the frame image as a NumPy array.
        """
        frame = typing.cast(dai.ImgFrame, self.rgb_queue.get())

        # update frame information
        self._manual_lens_pos = frame.getLensPosition()
        self._exposure = frame.getExposureTime()
        self._white_balance = frame.getColorTemperature()

        ts = int(frame.getTimestamp().total_seconds() * 1000)
        image = typing.cast(np.ndarray, frame.getCvFrame())

        self._last_rgb_frame = image
        self._last_ts = ts

    def release(self):
        """
        Releases the camera device, closing the connection and cleaning up resources.
        """
        self.device.__exit__(None, None, None)

    @property
    def gain(self) -> int:
        """
        Raises an exception indicating that gain adjustment is not supported.

        Raises:
            Exception: Gain is not supported.
        """
        raise Exception("Gain is not supported.")

    @gain.setter
    def gain(self, value: int):
        """
        Raises an exception indicating that gain adjustment is not supported.

        Raises:
            Exception: Gain is not supported.
        """
        raise Exception("Gain is not supported.")

    @property
    def iso(self) -> int:
        """
        Gets the ISO sensitivity setting for the camera.

        Returns:
            int: The current ISO sensitivity value.
        """
        return self._iso_sensitivity

    @iso.setter
    def iso(self, value: int):
        """
        Sets the ISO sensitivity for the camera, if the camera is running.

        Args:
            value (int): The ISO sensitivity value to set.
        """
        if not self.is_running:
            return

        self._iso_sensitivity = value

        # trigger exposure to set value
        self.exposure = self.exposure

    @property
    def exposure(self) -> int:
        """
        Gets the current exposure time in microseconds.

        Returns:
            int: The current exposure time in microseconds.
        """
        return int(self._exposure.total_seconds() * 1000 * 1000)

    @exposure.setter
    def exposure(self, value: int):
        """
        Sets the exposure time for the camera, if the camera is running.

        Args:
            value (int): The exposure time in microseconds to set.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        value = max(1, min(60 * 1000 * 1000, int(value)))
        self._exposure = timedelta(microseconds=value)
        ctrl.setManualExposure(self._exposure, self._iso_sensitivity)
        self.rgb_control_queue.send(ctrl)

    @property
    def enable_auto_exposure(self) -> bool:
        """
        Checks if auto exposure is enabled.

        Returns:
            bool: True if auto exposure is enabled, otherwise False.
        """
        return self._auto_exposure

    @enable_auto_exposure.setter
    def enable_auto_exposure(self, value: bool):
        """
        Enables or disables auto exposure for the camera, if the camera is running.

        Args:
            value (bool): Set to True to enable auto exposure, or False to disable.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        self._auto_exposure = value
        if value:
            ctrl.setAutoExposureEnable()
        else:
            ctrl.setManualExposure(self._exposure, self._iso_sensitivity)
        self.rgb_control_queue.send(ctrl)

    @property
    def enable_auto_white_balance(self) -> bool:
        """
        Checks if auto white balance is enabled.

        Returns:
            bool: True if auto white balance is enabled, otherwise False.
        """
        return self._auto_white_balance

    @enable_auto_white_balance.setter
    def enable_auto_white_balance(self, value: bool):
        """
        Enables or disables auto white balance for the camera, if the camera is running.

        Args:
            value (bool): Set to True to enable auto white balance, or False to disable.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        self._auto_white_balance = value
        if value:
            ctrl.setAutoWhiteBalanceMode(dai.RawCameraControl.AutoWhiteBalanceMode.AUTO)
        else:
            ctrl.setAutoWhiteBalanceMode(dai.RawCameraControl.AutoWhiteBalanceMode.OFF)
        self.rgb_control_queue.send(ctrl)

    @property
    def white_balance(self) -> int:
        """
        Gets the current white balance setting for the camera.

        Returns:
            int: The current white balance value.
        """
        return self._white_balance

    @white_balance.setter
    def white_balance(self, value: int):
        """
        Sets the white balance for the camera, if the camera is running.

        Args:
            value (int): The white balance value to set.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        value = max(1000, min(12000, int(value)))
        ctrl.setManualWhiteBalance(value)
        self.rgb_control_queue.send(ctrl)

    @property
    def auto_focus(self) -> bool:
        """
        Checks if auto focus is enabled.

        Returns:
            bool: True if auto focus is enabled, otherwise False.
        """
        return self._focus_mode == dai.RawCameraControl.AutoFocusMode.AUTO

    @auto_focus.setter
    def auto_focus(self, value: bool):
        """
        Enables or disables auto focus for the camera, if the camera is running.

        Args:
            value (bool): Set to True to enable auto focus, or False to disable.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        if value:
            self._focus_mode = dai.RawCameraControl.AutoFocusMode.AUTO
            ctrl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
        else:
            self._focus_mode = dai.RawCameraControl.AutoFocusMode.OFF
            ctrl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF)
        self.rgb_control_queue.send(ctrl)

    @property
    def focus_distance(self) -> int:
        """
        Gets the current manual focus distance setting for the camera.

        Returns:
            int: The current focus distance as an integer.
        """
        return self._manual_lens_pos

    @focus_distance.setter
    def focus_distance(self, position: int):
        """
        Sets the manual focus distance for the camera, if the camera is running.

        Args:
            position (int): The focus distance to set.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        position = max(0, min(255, int(position)))
        ctrl.setManualFocus(position)
        self.rgb_control_queue.send(ctrl)

    def get_camera_matrix(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        """
        Retrieves the camera intrinsic matrix for the specified stream type.

        Args:
            stream_type (CameraStreamType, optional): The type of camera stream (default is Color).

        Returns:
            np.ndarray: The intrinsic matrix as a NumPy array.
        """
        calibration_data = self.device.readCalibration()
        intrinsics = calibration_data.getCameraIntrinsics(self.color_board_socket)
        return np.array(intrinsics)

    def get_fisheye_distortion(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        """
        Retrieves the distortion coefficients for fisheye distortion for the specified stream type.

        Args:
            stream_type (CameraStreamType, optional): The type of camera stream (default is Color).

        Returns:
            np.ndarray: The distortion coefficients as a NumPy array.
        """
        calibration_data = self.device.readCalibration()
        distortion = calibration_data.getDistortionCoefficients(self.color_board_socket)
        return np.array(distortion)

    @property
    def serial(self) -> str:
        """
        Gets the serial number of the device.

        Returns:
            str: The serial number associated with the device.
        """
        info = self.device.getDeviceInfo()
        return info.mxid

    @property
    def camera_features(self) -> typing.List[CameraFeatures]:
        """
        Retrieves a list of connected camera features.

        Returns:
            List[CameraFeatures]: A list of features for the connected camera.
        """
        return self.device.getConnectedCameraFeatures()

    @property
    def device_info(self) -> dai.DeviceInfo:
        """
        Gets information about the device hardware.

        Returns:
            dai.DeviceInfo: The device information object containing hardware details.
        """
        return self.device.getDeviceInfo()

    @property
    def is_running(self):
        """
        Checks if the device pipeline is currently running.

        Returns:
            bool: True if the pipeline is running, otherwise False.
        """
        return self.device is not None and self.device.isPipelineRunning()
