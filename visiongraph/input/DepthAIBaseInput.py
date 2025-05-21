import typing
from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser, ArgumentError
from datetime import timedelta
from typing import Optional, Tuple

import cv2
import depthai as dai
import numpy as np
from depthai import CameraFeatures

from visiongraph.input.BaseCamera import BaseCamera
from visiongraph.model.CameraStreamType import CameraStreamType
from visiongraph.util import CommonArgs

_CameraProperties = dai.ColorCameraProperties


class DepthAIBaseInput(BaseCamera, ABC):
    """
    Abstract base class for DepthAI camera input handling.

    This class provides basic functionalities to manage camera properties, settings,
    and data streams for DepthAI-compatible cameras.
    """

    def __init__(self, mxid_or_name: Optional[str] = None):
        """
        Initializes the DepthAIBaseInput object with default settings for camera properties.

        :param mxid_or_name: MXID or IP/USB of the device.
        """
        super().__init__()

        # settings
        self.queue_max_size: int = 1

        self.color_sensor_resolution: dai.ColorCameraProperties.SensorResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P

        self.enable_color: bool = True
        self.enable_color_still: bool = False

        self.interleaved: bool = False
        self.color_isp_scale: Optional[Tuple[int, int]] = None
        self.color_board_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A
        self.color_fps: Optional[float] = None

        # settings
        self._focus_mode: dai.RawCameraControl.AutoFocusMode = dai.RawCameraControl.AutoFocusMode.AUTO
        self._manual_lens_pos: int = 0

        self._auto_exposure: bool = True
        self._auto_exposure_compensation: int = 0
        self._exposure: timedelta = timedelta(microseconds=30)
        self._iso_sensitivity: int = 400

        self._auto_white_balance: bool = True
        self._auto_white_balance_mode: dai.CameraControl.AutoWhiteBalanceMode = dai.CameraControl.AutoWhiteBalanceMode.AUTO
        self._white_balance: int = 1000

        self._anti_banding_mode: dai.CameraControl.AntiBandingMode = dai.CameraControl.AntiBandingMode.OFF
        self._effect_mode: dai.CameraControl.EffectMode = dai.CameraControl.EffectMode.OFF
        self._brightness: int = 0
        self._contrast: int = 0
        self._saturation: int = 0
        self._sharpness: int = 0
        self._luma_denoise: int = 0
        self._chroma_denoise: int = 0

        # device info
        self.mxid_or_name: Optional[str] = mxid_or_name
        self._initial_device_info: Optional[dai.DeviceInfo] = None

        # pipeline objects
        self.pipeline: Optional[dai.Pipeline] = None
        self.color_camera: Optional[dai.node.ColorCamera] = None
        self.device: Optional[dai.Device] = None
        self.color_still_encoder: Optional[dai.node.VideoEncoder] = None

        # node names
        self.rgb_stream_name = "rgb"
        self.rgb_isp_stream_name = "rgb_isp"
        self.rgb_control_in_name = "rbg_control_in"
        self.rgb_still_stream_name = "rgb_still"

        # nodes
        self.color_x_out: Optional[dai.node.XLinkOut] = None
        self.color_isp_out: Optional[dai.node.XLinkOut] = None
        self.color_still_out: Optional[dai.node.XLinkOut] = None
        self.color_control_in: Optional[dai.node.XLinkIn] = None

        self.rgb_control_queue: Optional[dai.DataInputQueue] = None
        self.rgb_queue: Optional[dai.DataOutputQueue] = None
        self.rgb_isp_queue: Optional[dai.DataOutputQueue] = None
        self.rgb_still_queue: Optional[dai.DataOutputQueue] = None

        # capture
        self._last_ts: int = 0
        self._last_rgb_frame: Optional[np.ndarray] = None
        self._last_rgb_still_frame: Optional[np.ndarray] = None

    def setup(self):
        """
        Sets up the camera pipeline and prepares the camera for streaming.

        This method initializes the pipeline, starts the device, and prepares the output queues for RGB streams.
        """
        self.pipeline = dai.Pipeline()

        if self.mxid_or_name is not None:
            self._initial_device_info = dai.DeviceInfo(self.mxid_or_name)

        self.pre_start_setup()

        # starts pipeline
        if self._initial_device_info is not None:
            device = dai.Device(self.pipeline, self._initial_device_info)
        else:
            device = dai.Device(self.pipeline)

        # camera starts - pre_start_setup is also called here
        self.device = device.__enter__()

        if self.enable_color:
            self.rgb_control_queue = self.device.getInputQueue(self.rgb_control_in_name)
            self.rgb_isp_queue = self.device.getOutputQueue(name=self.rgb_isp_stream_name, maxSize=self.queue_max_size,
                                                            blocking=False)
            self.rgb_queue = self.device.getOutputQueue(name=self.rgb_stream_name, maxSize=self.queue_max_size,
                                                        blocking=False)
            self.rgb_still_queue = self.device.getOutputQueue(name=self.rgb_still_stream_name,
                                                              maxSize=self.queue_max_size, blocking=False)

            # wait for the first isp frame
            rgb_isp_frame = typing.cast(dai.ImgFrame, self.rgb_isp_queue.get())
            self.width = rgb_isp_frame.getWidth()
            self.height = rgb_isp_frame.getHeight()

    def pre_start_setup(self):
        """
        Prepares the camera node and sets its properties before starting the streaming process.

        This method configures the camera parameters such as resolution, interleaved mode, and stream linking.
        """
        if self.enable_color:
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

            # setup still stream if needed
            if self.enable_color_still:
                self.color_still_encoder = self.pipeline.create(dai.node.VideoEncoder)
                self.color_still_encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)

                self.color_still_out = self.pipeline.create(dai.node.XLinkOut)
                self.color_still_out.setStreamName(self.rgb_still_stream_name)

                self.color_camera.still.link(self.color_still_encoder.input)
                self.color_still_encoder.bitstream.link(self.color_still_out.input)

    @abstractmethod
    def read(self) -> (int, Optional[np.ndarray]):
        """
        Reads the next RGB frame from the camera queue and updates internal properties.

        :return: The timestamp of the frame and the frame image as a NumPy array.
        """
        if self.enable_color:
            frame = typing.cast(dai.ImgFrame, self.rgb_queue.get())

            # update frame information
            self._manual_lens_pos = frame.getLensPosition()
            self._exposure = frame.getExposureTime()
            self._iso_sensitivity = frame.getSensitivity()
            self._white_balance = frame.getColorTemperature()

            ts = int(frame.getTimestamp().total_seconds() * 1000)
            image = typing.cast(np.ndarray, frame.getCvFrame())

            self._last_rgb_frame = image
            self._last_ts = ts

    def capture_color_still(self, post_processed: bool = True) -> Optional[np.ndarray]:
        """
        Captures a still image from the color camera if the device is running and still capture is enabled.

        This method sends a still image capture request to the camera and waits for the resulting frame.
        The captured image is decoded and stored for potential reuse.

        :returns: A decoded still image as a NumPy array if successful, otherwise None.
        """
        if not self.is_running:
            return None

        if not self.enable_color_still:
            return None

        # send capture request
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

        # wait for still available
        still_queue_blocking_state = self.rgb_still_queue.getBlocking()
        self.rgb_still_queue.setBlocking(True)

        raw_frame = typing.cast(dai.ImgFrame, self.rgb_still_queue.get())
        still_frame = cv2.imdecode(raw_frame.getData(), cv2.IMREAD_UNCHANGED)

        if post_processed:
            still_frame = self._post_process(0, still_frame)
        self._last_rgb_still_frame = still_frame

        self.rgb_still_queue.setBlocking(still_queue_blocking_state)
        return self._last_rgb_still_frame

    def release(self):
        """
        Releases the camera device, closing the connection and cleaning up resources.
        """
        self.device.__exit__(None, None, None)

    def configure(self, args: Namespace):
        """
        Configures the DepthAI input using command line arguments.

        :param args: The command line arguments to configure the input.
        """
        super().configure(args)

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds the DepthAI input parameters to the argument parser.

        :param parser: The argument parser to add parameters to.
        """
        super(DepthAIBaseInput, DepthAIBaseInput).add_params(parser)
        CommonArgs.add_source_argument(parser)

        try:
            parser.add_argument("--dai-id", default=None, type=str, help="DepthAI MXID or IP/USB of the device.")
        except ArgumentError as ex:
            if ex.message.startswith("conflicting"):
                return
            raise ex

    @property
    def gain(self) -> int:
        """
        Raises an exception indicating that gain adjustment is not supported.

        :raises Exception: Gain is not supported.
        """
        raise Exception("Gain is not supported.")

    @gain.setter
    def gain(self, value: int):
        """
        Raises an exception indicating that gain adjustment is not supported.

        :raises Exception: Gain is not supported.
        """
        raise Exception("Gain is not supported.")

    @property
    def iso(self) -> int:
        """
        Gets the ISO sensitivity setting for the camera.

        :return: The current ISO sensitivity value.
        """
        return self._iso_sensitivity

    @iso.setter
    def iso(self, value: int):
        """
        Sets the ISO sensitivity for the camera, if the camera is running.

        :param value: The ISO sensitivity value to set.
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

        :return: The current exposure time in microseconds.
        """
        return int(self._exposure.total_seconds() * 1000 * 1000)

    @exposure.setter
    def exposure(self, value: int):
        """
        Sets the exposure time for the camera, if the camera is running.

        :param value: The exposure time in microseconds to set.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        value = max(1, min(60 * 1000 * 1000, int(value)))
        self._exposure = timedelta(microseconds=value)
        ctrl.setManualExposure(self._exposure, self._iso_sensitivity)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def enable_auto_exposure(self) -> bool:
        """
        Checks if auto exposure is enabled.

        :return: True if auto exposure is enabled, otherwise False.
        """
        return self._auto_exposure

    @enable_auto_exposure.setter
    def enable_auto_exposure(self, value: bool):
        """
        Enables or disables auto exposure for the camera, if the camera is running.

        :param value: Set to True to enable auto exposure, or False to disable.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        self._auto_exposure = value
        if value:
            ctrl.setAutoExposureEnable()
        else:
            ctrl.setManualExposure(self._exposure, self._iso_sensitivity)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def enable_auto_white_balance(self) -> bool:
        """
        Checks if auto white balance is enabled.

        :return: True if auto white balance is enabled, otherwise False.
        """
        return self._auto_white_balance

    @enable_auto_white_balance.setter
    def enable_auto_white_balance(self, value: bool):
        """
        Enables or disables auto white balance for the camera, if the camera is running.

        :param value: Set to True to enable auto white balance, or False to disable.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        self._auto_white_balance = value
        if value:
            ctrl.setAutoWhiteBalanceMode(dai.RawCameraControl.AutoWhiteBalanceMode.AUTO)
        else:
            ctrl.setAutoWhiteBalanceMode(dai.RawCameraControl.AutoWhiteBalanceMode.OFF)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def white_balance(self) -> int:
        """
        Gets the current white balance setting for the camera.

        :return: The current white balance value.
        """
        return self._white_balance

    @white_balance.setter
    def white_balance(self, value: int):
        """
        Sets the white balance for the camera, if the camera is running.

        :param value: The white balance value to set.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        value = max(1000, min(12000, int(value)))
        ctrl.setManualWhiteBalance(value)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def auto_focus(self) -> bool:
        """
        Checks if auto focus is enabled.

        :return: True if auto focus is enabled, otherwise False.
        """
        return self._focus_mode == dai.RawCameraControl.AutoFocusMode.AUTO

    @auto_focus.setter
    def auto_focus(self, value: bool):
        """
        Enables or disables auto focus for the camera, if the camera is running.

        :param value: Set to True to enable auto focus, or False to disable.
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

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def focus_distance(self) -> int:
        """
        Gets the current manual focus distance setting for the camera.

        :return: The current focus distance as an integer.
        """
        return self._manual_lens_pos

    @focus_distance.setter
    def focus_distance(self, position: int):
        """
        Sets the manual focus distance for the camera, if the camera is running.

        :param position: The focus distance to set.
        """
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        position = max(0, min(255, int(position)))
        ctrl.setManualFocus(position)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def auto_exposure_compensation(self) -> int:
        """
        Gets the current auto exposure compensation value.

        :return: The current auto exposure compensation in the range [-9, 9].
        """
        return self._auto_exposure_compensation

    @auto_exposure_compensation.setter
    def auto_exposure_compensation(self, value: int):
        """
        Sets the auto exposure compensation for the camera, if the camera is running.

        :param value: Compensation value in the range [-9, 9].
        """
        if not self.is_running:
            return

        self._auto_exposure_compensation = max(-9, min(9, value))
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureCompensation(self._auto_exposure_compensation)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def anti_banding_mode(self) -> dai.CameraControl.AntiBandingMode:
        """
        Gets the current anti-banding mode.

        :return: The current anti-banding mode.
        """
        return self._anti_banding_mode

    @anti_banding_mode.setter
    def anti_banding_mode(self, mode: dai.CameraControl.AntiBandingMode):
        """
        Sets the anti-banding mode for the camera, if the camera is running.

        :param mode: Anti-banding mode value (e.g., dai.CameraControl.AntiBandingMode).
        """
        if not self.is_running:
            return

        self._anti_banding_mode = mode
        ctrl = dai.CameraControl()
        ctrl.setAntiBandingMode(mode)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def auto_white_balance_mode(self) -> dai.CameraControl.AutoWhiteBalanceMode:
        """
        Gets the current auto white balance mode.

        :return: The current AWB mode.
        """
        return self._auto_white_balance_mode

    @auto_white_balance_mode.setter
    def auto_white_balance_mode(self, mode: dai.CameraControl.AutoWhiteBalanceMode):
        """
        Sets the auto white balance mode for the camera, if the camera is running.

        :param mode: Auto white balance mode (e.g., dai.CameraControl.AutoWhiteBalanceMode).
        """
        if not self.is_running:
            return

        self._auto_white_balance_mode = mode
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(mode)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def effect_mode(self) -> dai.CameraControl.EffectMode:
        """
        Gets the current image effect mode.

        :return: The current effect mode.
        """
        return self._effect_mode

    @effect_mode.setter
    def effect_mode(self, mode: dai.CameraControl.EffectMode):
        """
        Sets the image effect mode for the camera, if the camera is running.

        :param mode: The image effect mode (e.g., dai.CameraControl.EffectMode).
        """
        if not self.is_running:
            return

        self._effect_mode = mode
        ctrl = dai.CameraControl()
        ctrl.setEffectMode(mode)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def brightness(self) -> int:
        """
        Gets the current brightness setting.

        :return: The brightness value in the range [-10, 10].
        """
        return self._brightness

    @brightness.setter
    def brightness(self, value: int):
        """
        Sets the brightness for the camera, if the camera is running.

        :param value: Brightness value in the range [-10, 10].
        """
        if not self.is_running:
            return

        self._brightness = max(-10, min(10, value))
        ctrl = dai.CameraControl()
        ctrl.setBrightness(self._brightness)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def contrast(self) -> int:
        """
        Gets the current contrast setting.

        :return: The contrast value in the range [-10, 10].
        """
        return self._contrast

    @contrast.setter
    def contrast(self, value: int):
        """
        Sets the contrast for the camera, if the camera is running.

        :param value: Contrast value in the range [-10, 10].
        """
        if not self.is_running:
            return

        self._contrast = max(-10, min(10, value))
        ctrl = dai.CameraControl()
        ctrl.setContrast(self._contrast)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def saturation(self) -> int:
        """
        Gets the current saturation setting.

        :return: The saturation value in the range [-10, 10].
        """
        return self._saturation

    @saturation.setter
    def saturation(self, value: int):
        """
        Sets the saturation for the camera, if the camera is running.

        :param value: Saturation value in the range [-10, 10].
        """
        if not self.is_running:
            return

        self._saturation = max(-10, min(10, value))
        ctrl = dai.CameraControl()
        ctrl.setSaturation(self._saturation)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def sharpness(self) -> int:
        """
        Gets the current sharpness setting.

        :return: The sharpness value in the range [0, 4].
        """
        return self._sharpness

    @sharpness.setter
    def sharpness(self, value: int):
        """
        Sets the sharpness for the camera, if the camera is running.

        :param value: Sharpness value in the range [0, 4].
        """
        if not self.is_running:
            return

        self._sharpness = max(0, min(4, value))
        ctrl = dai.CameraControl()
        ctrl.setSharpness(self._sharpness)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def luma_denoise(self) -> int:
        """
        Gets the current luma denoise setting.

        :return: The luma denoise value in the range [0, 4].
        """
        return self._luma_denoise

    @luma_denoise.setter
    def luma_denoise(self, value: int):
        """
        Sets the luma denoise for the camera, if the camera is running.

        :param value: Luma denoise value in the range [0, 4].
        """
        if not self.is_running:
            return

        self._luma_denoise = max(0, min(4, value))
        ctrl = dai.CameraControl()
        ctrl.setLumaDenoise(self._luma_denoise)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    @property
    def chroma_denoise(self) -> int:
        """
        Gets the current chroma denoise setting.

        :return: The chroma denoise value in the range [0, 4].
        """
        return self._chroma_denoise

    @chroma_denoise.setter
    def chroma_denoise(self, value: int):
        """
        Sets the chroma denoise for the camera, if the camera is running.

        :param value: Chroma denoise value in the range [0, 4].
        """
        if not self.is_running:
            return

        self._chroma_denoise = max(0, min(4, value))
        ctrl = dai.CameraControl()
        ctrl.setChromaDenoise(self._chroma_denoise)

        if self.rgb_control_queue is not None:
            self.rgb_control_queue.send(ctrl)

    def get_camera_matrix(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        """
        Retrieves the camera intrinsic matrix for the specified stream type.

        :param stream_type: The type of camera stream (default is Color).

        :return: The intrinsic matrix as a NumPy array.
        """
        calibration_data = self.device.readCalibration()
        intrinsics = calibration_data.getCameraIntrinsics(self.color_board_socket)
        return np.array(intrinsics)

    def get_fisheye_distortion(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        """
        Retrieves the distortion coefficients for fisheye distortion for the specified stream type.

        :param stream_type: The type of camera stream (default is Color).

        :return: The distortion coefficients as a NumPy array.
        """
        calibration_data = self.device.readCalibration()
        distortion = calibration_data.getDistortionCoefficients(self.color_board_socket)
        return np.array(distortion)

    @property
    def serial(self) -> str:
        """
        Gets the serial number of the device.

        :return: The serial number associated with the device.
        """
        info = self.device.getDeviceInfo()
        return info.mxid

    @property
    def camera_features(self) -> typing.List[CameraFeatures]:
        """
        Retrieves a list of connected camera features.

        :return: A list of features for the connected camera.
        """
        return self.device.getConnectedCameraFeatures()

    @property
    def device_info(self) -> dai.DeviceInfo:
        """
        Gets information about the device hardware.

        :return: The device information object containing hardware details.
        """
        return self.device.getDeviceInfo()

    @property
    def is_running(self):
        """
        Checks if the device pipeline is currently running.

        :return: True if the pipeline is running, otherwise False.
        """
        return self.device is not None and self.device.isPipelineRunning()
