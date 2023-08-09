import typing
from datetime import timedelta
from typing import Optional, Tuple

import depthai as dai
import numpy as np
from depthai import CameraFeatures

from visiongraph.input.BaseCamera import BaseCamera
from visiongraph.model.CameraStreamType import CameraStreamType

_CameraProperties = dai.ColorCameraProperties


class Oak1Input(BaseCamera):

    def __init__(self):
        super().__init__()

        # settings
        self.sensor_resolution: _CameraProperties.SensorResolution = _CameraProperties.SensorResolution.THE_1080_P

        self.interleaved: bool = False
        self.isp_scale: Optional[Tuple[int, int]] = None
        self.camera_board_socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A

        self._focus_mode: dai.RawCameraControl.AutoFocusMode = dai.RawCameraControl.AutoFocusMode.AUTO
        self._manual_lens_pos: int = 0

        self._auto_exposure: bool = True
        self._exposure: timedelta = timedelta(microseconds=30)
        self._iso_sensitivity: int = 400

        self._auto_white_balance: bool = True
        self._white_balance: int = 1000

        # pipeline objects
        self.pipeline: Optional[dai.Pipeline] = None
        self.cam: Optional[dai.node.ColorCamera] = None
        self.device: Optional[dai.Device] = None

        # node names
        self.rgb_stream_name = "rgb"
        self.isp_stream_name = "isp"
        self.control_in_name = "control_in"

        # nodes
        self.x_out: Optional[dai.node.XLinkOut] = None
        self.isp_out: Optional[dai.node.XLinkOut] = None
        self.control_in: Optional[dai.node.XLinkIn] = None

        self.control_queue: Optional[dai.DataInputQueue] = None
        self.rgb_queue: Optional[dai.DataOutputQueue] = None
        self.isp_queue: Optional[dai.DataOutputQueue] = None

    def setup(self):
        self.pipeline = dai.Pipeline()

        self.cam = self.pipeline.create(dai.node.ColorCamera)
        self.cam.setBoardSocket(self.camera_board_socket)
        self.cam.setResolution(self.sensor_resolution)
        # self.cam.setVideoSize(self.width, self.height)
        self.cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.cam.setInterleaved(self.interleaved)

        if self.isp_scale is not None:
            self.cam.setIspScale(self.isp_scale[0], self.isp_scale[1])

        self.x_out = self.pipeline.create(dai.node.XLinkOut)
        self.x_out.setStreamName(self.rgb_stream_name)
        self.cam.video.link(self.x_out.input)

        self.isp_out = self.pipeline.create(dai.node.XLinkOut)
        self.isp_out.setStreamName(self.isp_stream_name)
        self.cam.isp.link(self.isp_out.input)

        self.control_in = self.pipeline.create(dai.node.XLinkIn)
        self.control_in.setStreamName(self.control_in_name)
        self.control_in.out.link(self.cam.inputControl)

        # starts pipeline
        self.device = dai.Device(self.pipeline)

        self.control_queue = self.device.getInputQueue(self.control_in_name)
        self.rgb_queue = self.device.getOutputQueue(name=self.rgb_stream_name, maxSize=1, blocking=False)
        self.isp_queue = self.device.getOutputQueue(name=self.isp_stream_name, maxSize=1, blocking=False)

        isp_frame = typing.cast(dai.ImgFrame, self.isp_queue.get())
        self.width = isp_frame.getWidth()
        self.height = isp_frame.getHeight()

    def read(self) -> (int, Optional[np.ndarray]):
        frame = typing.cast(dai.ImgFrame, self.rgb_queue.get())

        # update frame information
        self._manual_lens_pos = frame.getLensPosition()
        self._exposure = frame.getExposureTime()
        self._white_balance = frame.getColorTemperature()

        ts = int(frame.getTimestamp().total_seconds() * 1000)
        image = typing.cast(np.ndarray, frame.getCvFrame())

        return self._post_process(ts, image)

    def release(self):
        self.device.close()

    @property
    def gain(self) -> int:
        raise Exception("Gain is not supported.")

    @gain.setter
    def gain(self, value: int):
        raise Exception("Gain is not supported.")

    @property
    def iso(self) -> int:
        return self._iso_sensitivity

    @iso.setter
    def iso(self, value: int):
        if not self.is_running:
            return

        self._iso_sensitivity = value

        # trigger exposure to set value
        self.exposure = self.exposure

    @property
    def exposure(self) -> int:
        return int(self._exposure.total_seconds() * 1000 * 1000)

    @exposure.setter
    def exposure(self, value: int):
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        value = max(1, min(60 * 1000 * 1000, int(value)))
        self._exposure = timedelta(microseconds=value)
        ctrl.setManualExposure(self._exposure, self._iso_sensitivity)
        self.control_queue.send(ctrl)

    @property
    def enable_auto_exposure(self) -> bool:
        return self._auto_exposure

    @enable_auto_exposure.setter
    def enable_auto_exposure(self, value: bool):
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        self._auto_exposure = value
        if value:
            ctrl.setAutoExposureEnable()
        else:
            ctrl.setManualExposure(self._exposure, self._iso_sensitivity)
        self.control_queue.send(ctrl)

    @property
    def enable_auto_white_balance(self) -> bool:
        return self._auto_white_balance

    @enable_auto_white_balance.setter
    def enable_auto_white_balance(self, value: bool):
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        self._auto_white_balance = value
        if value:
            ctrl.setAutoWhiteBalanceMode(dai.RawCameraControl.AutoWhiteBalanceMode.AUTO)
        else:
            ctrl.setAutoWhiteBalanceMode(dai.RawCameraControl.AutoWhiteBalanceMode.OFF)
        self.control_queue.send(ctrl)

    @property
    def white_balance(self) -> int:
        return self._white_balance

    @white_balance.setter
    def white_balance(self, value: int):
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        value = max(1000, min(12000, int(value)))
        ctrl.setManualWhiteBalance(value)
        self.control_queue.send(ctrl)

    @property
    def auto_focus(self) -> bool:
        return self._focus_mode == dai.RawCameraControl.AutoFocusMode.AUTO

    @auto_focus.setter
    def auto_focus(self, value: bool):
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
        self.control_queue.send(ctrl)

    @property
    def focus_distance(self) -> int:
        return self._manual_lens_pos

    @focus_distance.setter
    def focus_distance(self, position: int):
        if not self.is_running:
            return

        ctrl = dai.CameraControl()
        position = max(0, min(255, int(position)))
        ctrl.setManualFocus(position)
        self.control_queue.send(ctrl)

    def get_camera_matrix(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        calibration_data = self.device.readCalibration()
        intrinsics = calibration_data.getCameraIntrinsics(self.camera_board_socket)
        return np.array(intrinsics)

    def get_fisheye_distortion(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        calibration_data = self.device.readCalibration()
        distortion = calibration_data.getDistortionCoefficients(self.camera_board_socket)
        return np.array(distortion)

    @property
    def serial(self) -> str:
        info = self.device.getDeviceInfo()
        return info.mxid

    @property
    def camera_features(self) -> typing.List[CameraFeatures]:
        return self.device.getConnectedCameraFeatures()

    @property
    def device_info(self) -> dai.DeviceInfo:
        return self.device.getDeviceInfo()

    @property
    def is_running(self):
        return self.device is not None and self.device.isPipelineRunning()
