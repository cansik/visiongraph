import json
import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, List, Any

import numpy as np
import pyrealsense2 as rs
import vector

from visiongraph.input.BaseDepthCamera import BaseDepthCamera
from visiongraph.model.CameraStreamType import CameraStreamType
from visiongraph.model.geometry.BoundingBox2D import BoundingBox2D
from visiongraph.model.types.RealSenseColorScheme import RealSenseColorScheme
from visiongraph.model.types.RealSenseFilter import RealSenseFilters
from visiongraph.util import ImageUtils, CommonArgs
from visiongraph.util.ArgUtils import add_enum_choice_argument, add_dict_choice_argument
from visiongraph.util.TimeUtils import current_millis


class RealSenseInput(BaseDepthCamera):
    """
    A class to interface with Intel RealSense depth cameras for capturing depth, infrared and color streams.
    """

    def __init__(self):
        """
        Initializes the RealSenseInput object with default settings and parameters for the RealSense camera.
        """
        super().__init__()

        self.disable_emitter = False
        self.selected_serial: Optional[str] = None

        self.input_bag_file: Optional[str] = None
        self.output_bag_file: Optional[str] = None

        self.colorizer: Optional[rs.colorizer] = None
        self.color_scheme = RealSenseColorScheme.WhiteToBlack

        self.pipeline: Optional[rs.pipeline] = None
        self.frames: Optional[rs.composite_frame] = None
        self.align: Optional[rs.align] = None

        self.profile: Optional[rs.pipeline_profile] = None
        self.device: Optional[rs.device] = None
        self.image_sensor: Optional[rs.sensor] = None

        self._depth_frame: Optional[rs.depth_frame] = None

        self.color_format: rs.format = rs.format.bgr8
        self.depth_format: rs.format = rs.format.z16

        self.infrared_width: Optional[int] = None
        self.infrared_height: Optional[int] = None
        self.infrared_format: rs.format = rs.format.y8

        self.play_any_bag_stream = True
        self.bag_offline_playback = True

        self.json_config_path: Optional[str] = None

        self.auto_exposure_limit: Optional[int] = None
        self.auto_gain_limit: Optional[int] = None

        # filter
        self.depth_filters: List[rs.filter] = []
        self._filters_to_enable: List[type(rs.filter)] = []

        self.config: Optional[rs.config] = None

        self.frame_read_timeout: int = 5000

    def setup(self):
        """
        Prepares the RealSense camera for operation by setting up streams and configuring parameters.
        """
        ctx = rs.context()

        if self.device_count == 0 and self.input_bag_file is None:
            raise Exception("No RealSense device found!")

        if self.input_bag_file is not None and self.play_any_bag_stream:
            self.allow_any_stream()

        # update dimension for different inputs
        if self.depth_width is None:
            self.depth_width = self.width

        if self.depth_height is None:
            self.depth_height = self.height

        if self.infrared_width is None:
            self.infrared_width = self.width

        if self.infrared_height is None:
            self.infrared_height = self.height

        self.pipeline = rs.pipeline(ctx)

        self.config = rs.config() if self.config is None else self.config

        if self.selected_serial is not None:
            self.config.enable_device(serial=self.selected_serial)

        if self.input_bag_file is not None:
            rs.config.enable_device_from_file(self.config, self.input_bag_file)
            rs.config.enable_all_streams(self.config)

        if self.output_bag_file is not None:
            self.config.enable_record_to_file(self.output_bag_file)

        if self.use_infrared:
            self.config.enable_stream(rs.stream.infrared, self.infrared_width, self.infrared_height,
                                      self.infrared_format, int(self.fps))
            self.align = rs.align(rs.stream.infrared)
        else:
            self.config.enable_stream(rs.stream.color, self.width, self.height, self.color_format, int(self.fps))
            self.align = rs.align(rs.stream.color)

        if self.enable_depth:
            self.colorizer = rs.colorizer(color_scheme=self.color_scheme.value)
            self.config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height,
                                      self.depth_format, int(self.fps))
            [self.depth_filters.append(f()) for f in self._filters_to_enable]

        # set options before startup (only for live-camera feed)
        if self.input_bag_file is None:
            device = self._find_current_device(ctx, self.selected_serial)
            depth_sensor: rs.depth_stereo_sensor = device.first_depth_sensor()

            if depth_sensor is not None:
                def get_option_max_or_value(option: rs.option, value: Optional[Any]) -> float:
                    if not depth_sensor.supports(option):
                        logging.warning(f"The option {option} is not supported!")
                        return value

                    if value is not None:
                        return float(value)

                    option_range: rs.option_range = depth_sensor.get_option_range(option)
                    return option_range.max

                try:
                    self.set_option(rs.option.auto_exposure_limit, sensor=depth_sensor,
                                    value=get_option_max_or_value(rs.option.auto_exposure_limit,
                                                                  self.auto_exposure_limit))
                except Exception as ex:
                    logging.error(f"Could not set auto_exposure_limit: {ex}")

                try:
                    self.set_option(rs.option.auto_gain_limit, sensor=depth_sensor,
                                    value=get_option_max_or_value(rs.option.auto_gain_limit, self.auto_gain_limit))
                except Exception as ex:
                    logging.error(f"Could not set auto_gain_limit: {ex}")

        # start up device
        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()
        depth_sensor = self.device.first_depth_sensor()

        # set emitter state
        self.set_option(rs.option.emitter_enabled, 0 if self.disable_emitter else 1, depth_sensor)

        # set default image sensor
        self.image_sensor = self.device.first_depth_sensor() if self.use_infrared else self.device.first_color_sensor()

        # applying other options
        try:
            self._apply_initial_settings()
        except Exception as ex:
            logging.warning(f"Could not apply initial RealSense settings: {ex}")

        # apply json config
        if self.json_config_path is not None:
            self.load_json_config_from_file(self.json_config_path)

        # set playback options
        if self.device.is_playback():
            playback: rs.playback = self.profile.get_device().as_playback()
            playback.set_real_time(not self.bag_offline_playback)

    def release(self):
        """
        Releases the RealSense camera and stops the pipeline.
        """
        self.pipeline.stop()

    def read(self) -> (int, Optional[np.ndarray]):
        """
        Reads the next frame from the camera.

        """
        success, self.frames = self.pipeline.try_wait_for_frames(timeout_ms=self.frame_read_timeout)
        time_stamp = current_millis()

        if not success:
            if self.device.is_playback():
                success, self.frames = self.pipeline.try_wait_for_frames(timeout_ms=self.frame_read_timeout)
                if not success:
                    raise Exception("RealSense: Bag frame could not be read from device.")
                else:
                    logging.warning("Skipping bag file frame")
            else:
                raise Exception("RealSense: Frame could not be read from device.")

        if self.align is not None:
            # alignment only happens if depth is enabled!
            self.frames = self.align.process(self.frames)

        # filter depth
        if self.enable_depth:
            self._depth_frame = self.frames.get_depth_frame()

            for depth_filter in self.depth_filters:
                self._depth_frame = depth_filter.process(self._depth_frame).as_depth_frame()

        if self.use_infrared:
            image = self.frames.get_infrared_frame()
        else:
            image = self.frames.get_color_frame()

        if self.use_depth_as_input:
            return self._post_process(time_stamp, self.depth_map)

        if image is None:
            logging.warning("could not read frame.")
            return self._post_process(time_stamp, None)

        return self._post_process(time_stamp, np.asanyarray(image.get_data()))

    @property
    def depth_frame(self):
        """
        :return: The current depth frame from the RealSense camera.

        :raises Exception: If depth is not enabled for RealSense input.
        """
        if self._depth_frame is None:
            raise Exception("Depth is not enabled for RealSense input.")

        return self._depth_frame

    def _find_current_device(self, ctx: rs.context, serial: Optional[str] = None) -> rs.device:
        """
        Finds the current RealSense device.

        :param ctx: The RealSense context.
        :param serial: The serial number of the device to find.

        :return: The device found.

        :raises Exception: If no RealSense device is connected or specified device cannot be found.
        """
        devices: List[rs.device] = ctx.devices

        if len(devices) == 0:
            raise Exception("No RealSense device is connected.")

        for device in devices:
            if serial is None:
                return device

            device_serial = device.get_info(rs.camera_info.serial_number)
            if device_serial == serial:
                return device

        raise Exception(f"Device with serial {serial} could not be found!")

    def distance(self, x: float, y: float) -> float:
        """
        Computes the distance to the point (x, y) in the depth frame.

        :param x: The x-coordinate in the depth frame.
        :param y: The y-coordinate in the depth frame.

        :return: The distance to the point in meters.
        """
        depth_frame = self.depth_frame
        ix, iy = self._calculate_depth_coordinates(x, y, depth_frame.width, depth_frame.height)

        return depth_frame.get_distance(ix, iy)

    def pixel_to_point(self, x: float, y: float, depth_kernel_size: int = 1) -> vector.Vector3D:
        """
        Converts pixel coordinates to a 3D point in space.

        :param x: The x-coordinate of the pixel.
        :param y: The y-coordinate of the pixel.
        :param depth_kernel_size: The size of the kernel for depth averaging.

        :return: The corresponding 3D point in space.
        """
        depth_frame: rs.depth_frame = self.depth_frame
        ix, iy = self._calculate_depth_coordinates(x, y, depth_frame.width, depth_frame.height)

        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        if depth_kernel_size == 1:
            distance = depth_frame.get_distance(ix, iy)
        else:
            depth_data = np.asarray(self.depth_frame.data, dtype=float) * depth_frame.get_units()
            roi = ImageUtils.roi(depth_data, BoundingBox2D.from_kernel(ix, iy, depth_kernel_size))
            distance = np.median(roi)

        point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [ix, iy], distance)
        return vector.obj(x=point[0], y=point[1], z=point[2])

    @property
    def depth_map(self) -> np.ndarray:
        """
        :return: The depth map as a colorized numpy array.
        """
        depth_frame = self.depth_frame
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return depth_colormap

    @property
    def depth_buffer(self) -> np.ndarray:
        """
        :return: The raw depth data as a numpy array.
        """
        return np.asarray(self.depth_frame.data, dtype=float)

    def allow_any_stream(self):
        """
        Configures the input to allow any stream settings (width, height, fps).
        """
        self.width = 0
        self.height = 0
        self.fps = 0
        self.infrared_width = 0
        self.infrared_height = 0
        self.depth_width = 0
        self.depth_height = 0
        self.color_format = rs.format.any
        self.depth_format = rs.format.any
        self.infrared_format = rs.format.any

    def load_json_config_from_file(self, json_path: str):
        """
        Loads the JSON configuration from a specified file.

        :param json_path: The path to the JSON configuration file.
        """
        json_config = json.load(open(json_path, "r"))
        self.load_json_config(json_config)

    def load_json_config(self, json_config: str):
        """
        Applies the given JSON configuration to the RealSense device.

        :param json_config: The JSON configuration data as a string.
        """
        if self.device is None:
            logging.warning(f"No device available to apply json config.")
            return

        if not self.device.supports(rs.camera_info.advanced_mode):
            logging.warning(f"Device {self.device_name} does not support serialisation.")
            return

        serdev = rs.serializable_device(self.device)

        json_config = str(json_config).replace("'", '\"')
        serdev.load_json(json_config)

        logging.info(f"Json config has been loaded {self.json_config_path}")

    def get_json_config(self) -> str:
        """
        Serializes the current configuration of the RealSense device to a JSON string.

        :return: The serialized JSON configuration of the device.
        """
        if self.device is None:
            logging.warning(f"No device available to apply json config.")
            return ""

        if not self.device.supports(rs.camera_info.advanced_mode):
            logging.warning(f"Device {self.device_name} does not support serialisation.")
            return ""

        serdev = rs.serializable_device(self.device)
        return serdev.serialize_json()

    def get_realsense_intrinsics(self, stream_type: Optional[rs.stream] = None,
                                 stream_index: int = -1) -> rs.intrinsics:
        """
        Retrieves the intrinsics of the selected stream type.

        :param stream_type: The type of stream to get intrinsics for.
        :param stream_index: The index of the stream, defaults to -1.

        :return: The camera intrinsics for the specified stream.
        """
        profiles = self.pipeline.get_active_profile()

        # determine main stream_type type
        if stream_type is None:
            if self.use_infrared:
                stream_type = rs.stream.infrared
            else:
                stream_type = rs.stream.color
            logging.info(f"determined {stream_type} intrinsics")

        stream = profiles.get_stream(stream_type, stream_index).as_video_stream_profile()
        intrinsics: rs.intrinsics = stream.get_intrinsics()
        return intrinsics

    @staticmethod
    def _to_rs2_stream_type(stream: CameraStreamType) -> rs.stream:
        """
        Converts a CameraStreamType to the corresponding RealSense stream type.

        :param stream: The stream type to convert.

        :return: The corresponding RealSense stream type.

        :raises Exception: If the provided stream type is not available.
        """
        if stream == CameraStreamType.Color:
            return rs.stream.color
        elif stream == CameraStreamType.Depth:
            return rs.stream.depth
        elif stream == CameraStreamType.Infrared:
            return rs.stream.infrared

        raise Exception(f"RealSense stream type {stream} not available.")

    def get_camera_matrix(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        """
        Retrieves the camera matrix for the specified stream type.

        :param stream_type: The type of stream (default is Color).

        :return: The camera matrix as a 3x3 numpy array.
        """
        intrinsics = self.get_realsense_intrinsics(self._to_rs2_stream_type(stream_type))
        return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                         [0, intrinsics.fy, intrinsics.ppy],
                         [0, 0, 1]])

    def get_fisheye_distortion(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        """
        Retrieves the distortion coefficients for the specified stream type.

        :param stream_type: The type of stream (default is Color).

        :return: The distortion coefficients as a numpy array.
        """
        intrinsics = self.get_realsense_intrinsics(self._to_rs2_stream_type(stream_type))
        return np.array(intrinsics.coeffs[:4])

    def pre_process_image(self, image: np.ndarray,
                          stream_type: CameraStreamType = CameraStreamType.Color) -> Optional[np.ndarray]:
        """
        Preprocesses the given image based on the stream type.

        :param image: The image to preprocess.
        :param stream_type: The type of stream (default is Color).

        :return: The preprocessed image or None.
        """
        if stream_type == CameraStreamType.Depth:
            return np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())

        return image

    def get_raw_image(self, stream_type: CameraStreamType = CameraStreamType.Color) -> Optional[np.ndarray]:
        """
        Retrieves the raw image data for the specified stream type.

        :param stream_type: The type of stream (default is Color).

        :return: The raw image data as a numpy array or None.
        """
        if stream_type == CameraStreamType.Depth:
            return self.depth_buffer
        elif stream_type == CameraStreamType.Infrared:
            return np.asanyarray(self.frames.get_infrared_frame().get_data())
        elif stream_type == CameraStreamType.Color:
            return np.asanyarray(self.frames.get_color_frame().get_data())

        return None

    @property
    def device_count(self) -> int:
        """
        :return: The number of connected RealSense devices.
        """
        ctx = rs.context()
        return len(ctx.query_devices())

    def get_option(self, option: rs.option, sensor: Optional[rs.sensor] = None) -> float:
        """
        Retrieves the value of the specified option for the given sensor.

        :param option: The option to retrieve.
        :param sensor: The sensor for which to get the option value.

        :return: The value of the option.

:return: 
        """
        if sensor is None:
            sensor = self.image_sensor

        if sensor is None:
            logging.warning(f"No sensor for option {option} available!")
            return 0.0

        if sensor.supports(option):
            return sensor.get_option(option)
        else:
            logging.warning(f"The option {option} is not supported!")
            return 0.0

    def set_option(self, option: rs.option, value: float, sensor: Optional[rs.sensor] = None):
        """
        Sets the specified option for the given sensor.

        :param option: The option to set.
        :param value: The value to set for the option.
        :param sensor: The sensor for which to set the option value.

:param Notes: 
        """
        if sensor is None:
            sensor = self.image_sensor

        if sensor is None:
            logging.warning(f"No sensor for option {option} available!")
            return 0.0

        if sensor.supports(option):
            if sensor.is_option_read_only(option):
                logging.warning(f"The option {option} is read-only!")
                return

            sensor.set_option(option, float(value))
        else:
            logging.warning(f"The option {option} is not supported!")

    @property
    def device_name(self) -> str:
        """
        :return: The name of the RealSense device.
        """
        if self.device is None:
            return "NoDevice"
        return self.device.get_info(rs.camera_info.name)

    @property
    def gain(self) -> int:
        """
        :return: The current gain setting for the device.
        """
        return int(self.get_option(rs.option.gain))

    @gain.setter
    def gain(self, value: int):
        self.set_option(rs.option.gain, value)

    @property
    def exposure(self) -> int:
        """
        :return: The current exposure setting for the device.
        """
        return int(self.get_option(rs.option.exposure))

    @exposure.setter
    def exposure(self, value: int):
        self.set_option(rs.option.exposure, value)

    @property
    def enable_auto_exposure(self) -> bool:
        """
        :return: Whether auto exposure is enabled.
        """
        return bool(self.get_option(rs.option.enable_auto_exposure))

    @enable_auto_exposure.setter
    def enable_auto_exposure(self, value: bool):
        self.set_option(rs.option.enable_auto_exposure, value)

    @property
    def enable_auto_white_balance(self) -> bool:
        """
        :return: Whether auto white balance is enabled.
        """
        return bool(self.get_option(rs.option.enable_auto_white_balance))

    @enable_auto_white_balance.setter
    def enable_auto_white_balance(self, value: bool):
        self.set_option(rs.option.enable_auto_white_balance, value)

    @property
    def white_balance(self) -> int:
        """
        :return: The current white balance setting for the device.
        """
        return int(self.get_option(rs.option.white_balance))

    @white_balance.setter
    def white_balance(self, value: int):
        value = value // 100 * 100
        self.set_option(rs.option.white_balance, value)

    @property
    def serial(self) -> str:
        """
        :return: The serial number of the RealSense device.
        """
        return str(self.device.get_info(rs.camera_info.serial_number))

    def configure(self, args: Namespace):
        """
        Configures the RealSense input object based on provided command line arguments.

        :param args: The command line arguments.
        """
        super().configure(args)

        if args.source is not None:
            args.rs_play_bag = args.source

        self.selected_serial = args.rs_serial

        self.input_bag_file = args.rs_play_bag
        self.bag_offline_playback = args.rs_bag_offline
        self.output_bag_file = args.rs_record_bag

        self.disable_emitter = args.rs_disable_emitter
        self.color_scheme = args.rs_color_scheme

        self.json_config_path = args.rs_json

        if args.rs_auto_exposure_limit is not None:
            self.auto_exposure_limit = int(args.rs_auto_exposure_limit)

        if args.rs_auto_gain_limit is not None:
            self.auto_gain_limit = int(args.rs_auto_gain_limit)

        # filter enabler
        if args.rs_filter is not None:
            self._filters_to_enable = args.rs_filter

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command line argument parameters for configuring RealSense input.

        :param parser: The ArgumentParser instance to add parameters to.
        """
        super(RealSenseInput, RealSenseInput).add_params(parser)

        CommonArgs.add_source_argument(parser)

        parser.add_argument("--rs-serial", default=None, type=str,
                            help="RealSense serial number to choose specific device.")
        parser.add_argument("--rs-json", default=None, type=str,
                            help="RealSense json configuration to apply.")
        parser.add_argument("--rs-play-bag", default=None, type=str,
                            help="Path to a pre-recorded bag file for playback.")
        parser.add_argument("--rs-record-bag", default=None, type=str,
                            help="Path to a bag file to store the current recording.")
        parser.add_argument("--rs-disable-emitter", action="store_true",
                            help="Disable RealSense IR emitter.")
        parser.add_argument("--rs-bag-offline", action="store_true",
                            help="Disable realtime bag playback.")
        parser.add_argument("--rs-auto-exposure-limit", default=None, type=int, help="Auto exposure limit (ms).")
        parser.add_argument("--rs-auto-gain-limit", default=None, type=int, help="Auto gain limit (16-248).")
        add_dict_choice_argument(parser, RealSenseFilters, "--rs-filter", help="RealSense depth filter",
                                 default=None, nargs="+")
        add_enum_choice_argument(parser, RealSenseColorScheme, "--rs-color-scheme",
                                 default=RealSenseColorScheme.WhiteToBlack,
                                 help="Color scheme for depth map")
