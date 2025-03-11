import typing
from argparse import Namespace, ArgumentParser
from enum import Enum
from typing import Optional

import cv2
import depthai as dai
import numpy as np

from visiongraph.input.BaseDepthCamera import BaseDepthCamera
from visiongraph.input.DepthAIBaseInput import DepthAIBaseInput
from visiongraph.model.CameraStreamType import CameraStreamType


class OakDFrameAlignment(Enum):
    Disabled = 0
    Color = 1
    Infrared = 2


class OakDInput(DepthAIBaseInput, BaseDepthCamera):
    """
    A class to handle input from the Oak-D camera, managing both infrared
    and depth camera functionalities.
    """

    def __init__(self, mxid_or_name: Optional[str] = None):
        """
        Initializes the OakDInput object, setting up camera properties and
        internal states.

        :param mxid_or_name: MXID or IP/USB of the device.
        """
        super().__init__(mxid_or_name=mxid_or_name)

        self.color_sensor_resolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P

        self._ir_laser_dot_projector_intensity: float = 0  # 0..1
        self._ir_flood_light_intensity: float = 0  # 0..1

        self.select_ir_camera: dai.CameraBoardSocket = dai.CameraBoardSocket.LEFT

        self.ir_left_camera: Optional[dai.node.MonoCamera] = None
        self.ir_right_camera: Optional[dai.node.MonoCamera] = None
        self.active_ir_camera: Optional[dai.node.MonoCamera] = None
        self.ir_sensor_resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

        self.depth_node: Optional[dai.node.StereoDepth] = None
        self.depth_preset_mode: dai.node.StereoDepth.PresetMode = dai.node.StereoDepth.PresetMode.HIGH_DENSITY
        self.depth_median_filter: dai.MedianFilter = dai.MedianFilter.KERNEL_7x7
        self.depth_left_right_check: bool = True  # better handling for occlusions
        self.depth_subpixel: bool = False  # better accuracy for longer distance, fractional disparity 32-levels
        # closer-in minimum depth, disparity range is doubled (from 95 to 190)
        self.depth_extended_disparity: bool = False

        self.frame_alignment: OakDFrameAlignment = OakDFrameAlignment.Color

        # node names
        self.ir_stream_name = "ir"
        self.depth_stream_name = "depth"

        # nodes
        self.ir_x_out: Optional[dai.node.XLinkOut] = None
        self.depth_x_out: Optional[dai.node.XLinkOut] = None

        self.ir_queue: Optional[dai.DataOutputQueue] = None
        self.depth_queue: Optional[dai.DataOutputQueue] = None

        # capture
        self._last_ir_frame: Optional[np.ndarray] = None
        self._last_depth_frame: Optional[np.ndarray] = None

    def pre_start_setup(self):
        """
        Performs setup procedures before starting the camera pipeline,
        such as enabling depth and configuring infrared settings.
        """
        if self.use_depth_as_input:
            self.enable_depth = True

        super().pre_start_setup()

        # setup ir camera's
        if self.use_infrared or self.enable_depth:
            self.ir_left_camera = self.pipeline.create(dai.node.MonoCamera)
            self.ir_left_camera.setBoardSocket(dai.CameraBoardSocket.LEFT)
            self.ir_left_camera.setResolution(self.ir_sensor_resolution)

            self.ir_right_camera = self.pipeline.create(dai.node.MonoCamera)
            self.ir_right_camera.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            self.ir_right_camera.setResolution(self.ir_sensor_resolution)

            if self.select_ir_camera.LEFT:
                self.active_ir_camera = self.ir_left_camera
            else:
                self.active_ir_camera = self.ir_right_camera

            # link active ir camera
            self.ir_x_out = self.pipeline.create(dai.node.XLinkOut)
            self.ir_x_out.setStreamName(self.ir_stream_name)
            self.active_ir_camera.out.link(self.ir_x_out.input)

        # set depth camera settings
        if self.enable_depth:
            self.depth_node = self.pipeline.create(dai.node.StereoDepth)
            self.depth_node.setDefaultProfilePreset(self.depth_preset_mode)
            self.depth_node.initialConfig.setMedianFilter(self.depth_median_filter)
            self.depth_node.setLeftRightCheck(self.depth_left_right_check)
            self.depth_node.setExtendedDisparity(self.depth_extended_disparity)
            self.depth_node.setSubpixel(self.depth_subpixel)

            # setup depth align
            if self.frame_alignment == self.frame_alignment.Infrared:
                self.depth_node.setDepthAlign(camera=self.active_ir_camera.getBoardSocket())
            elif self.frame_alignment == self.frame_alignment.Color:
                self.depth_node.setDepthAlign(camera=self.color_camera.getBoardSocket())

            # link depth
            self.ir_left_camera.out.link(self.depth_node.left)
            self.ir_right_camera.out.link(self.depth_node.right)

            self.depth_x_out = self.pipeline.create(dai.node.XLinkOut)
            self.depth_x_out.setStreamName(self.depth_stream_name)
            self.depth_node.depth.link(self.depth_x_out.input)

    def setup(self):
        """
        Initializes output queues and camera settings after the pipeline is ready.
        """
        super().setup()

        if self.use_infrared:
            self.ir_queue = self.device.getOutputQueue(name=self.ir_stream_name,
                                                       maxSize=self.queue_max_size,
                                                       blocking=False)

        if self.enable_depth:
            self.depth_queue = self.device.getOutputQueue(name=self.depth_stream_name,
                                                          maxSize=self.queue_max_size,
                                                          blocking=False)

        self.device.setIrLaserDotProjectorIntensity(self._ir_laser_dot_projector_intensity)
        self.device.setIrFloodLightIntensity(self._ir_flood_light_intensity)

    def read(self) -> (int, Optional[np.ndarray]):
        """
        Reads the most recent infrared and depth frames from the respective queues.

        :return: A tuple containing the timestamp and the image
        """
        super().read()

        if self.use_infrared:
            ir_frame = typing.cast(dai.ImgFrame, self.ir_queue.get())
            ir_image = typing.cast(np.ndarray, ir_frame.getCvFrame())
            self._last_ir_frame = ir_image

        if self.enable_depth:
            depth_frame = typing.cast(dai.ImgFrame, self.depth_queue.get())
            depth_image = typing.cast(np.ndarray, depth_frame.getCvFrame())
            self._last_depth_frame = depth_image

        if self.use_depth_as_input:
            return self._post_process(self._last_ts, self.depth_map)

        if self.use_infrared:
            return self._post_process(self._last_ts, self._last_ir_frame)

        return self._post_process(self._last_ts, self._last_rgb_frame)

    def distance(self, x: float, y: float) -> float:
        """
        Calculates the distance in meters from the camera to a certain point using the depth data.

        :param x: The x-coordinate in the image.
        :param y: The y-coordinate in the image.

        :return: The distance in meters, or -1 if the device is not initialized.
        """
        if self.device is None:
            return -1

        depth_frame = self._last_depth_frame

        if self._last_depth_frame is None:
            return -1

        h, w = depth_frame.shape[:2]
        ix, iy = self._calculate_depth_coordinates(x, y, w, h)

        depth_value = float(depth_frame[iy, ix])

        return depth_value / 1000

    @property
    def depth_buffer(self) -> np.ndarray:
        """
        Provides access to the last captured depth frame.

        :return: The last depth frame.
        """
        return self._last_depth_frame

    @property
    def depth_map(self) -> np.ndarray:
        """
        Generates a color-mapped depth representation for visualization.

        :return: The colorized depth map.
        """
        dmap = self._colorize(self.depth_buffer, (0, 12000), cv2.COLORMAP_JET)
        return dmap

    @property
    def ir_laser_dot_projector_intensity(self):
        """
        Gets the intensity of the infrared laser dot projector.

        :return: The current intensity level.
        """
        return self._ir_laser_dot_projector_intensity

    @ir_laser_dot_projector_intensity.setter
    def ir_laser_dot_projector_intensity(self, value: int):
        """
        Sets the intensity of the infrared laser dot projector.

        :param value: The desired intensity level.
        """
        if self.device is not None:
            self.device.setIrLaserDotProjectorIntensity(value)
            self._ir_laser_dot_projector_intensity = value

    @property
    def ir_flood_light_intensity(self):
        """
        Gets the intensity of the infrared flood light.

        :return: The current intensity level.
        """
        return self._ir_laser_dot_projector_intensity

    @ir_flood_light_intensity.setter
    def ir_flood_light_intensity(self, value: int):
        """
        Sets the intensity of the infrared flood light.

        :param value: The desired intensity level.
        """
        if self.device is not None:
            self.device.setIrFloodLightIntensity(value)
            self._ir_flood_light_intensity = value

    def pre_process_image(self, image: np.ndarray,
                          stream_type: CameraStreamType = CameraStreamType.Color) -> Optional[np.ndarray]:
        """
        Pre-processes the input image based on the specified stream type.

        :param image: The raw image to be processed.
        :param stream_type: The type of camera stream (default is Color).

        :return: The processed image, or None if no processing is needed.
        """
        if stream_type == CameraStreamType.Depth:
            return self._colorize(image, (0, 12000), cv2.COLORMAP_JET)

        return image

    def get_raw_image(self, stream_type: CameraStreamType = CameraStreamType.Color) -> Optional[np.ndarray]:
        """
        Retrieves the raw image data based on the specified stream type.

        :param stream_type: The type of camera stream (default is Color).

        :return: The raw image data, or None if the stream type is invalid.
        """
        if stream_type == CameraStreamType.Depth:
            return self.depth_buffer
        elif stream_type == CameraStreamType.Infrared:
            return self._last_ir_frame
        elif stream_type == CameraStreamType.Color:
            return self._last_rgb_frame

        return None

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds the DepthAI input parameters to the argument parser.

        :param parser: The argument parser to add parameters to.
        """
        super(OakDInput, OakDInput).add_params(parser)
        parser.add_argument("--dai-disable-color", action="store_true",
                            help="Disables the color stream of the OAK-D.")

    def configure(self, args: Namespace):
        """
        Configures the OakDInput settings based on the provided command-line arguments.

        :param args: The command-line arguments for configuration.
        """
        super().configure(args)

        if self.use_infrared:
            self.frame_alignment = OakDFrameAlignment.Infrared

        self.enable_color = not args.dai_disable_color
