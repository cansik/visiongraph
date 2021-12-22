import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, List, Tuple

import numpy as np
import pyrealsense2 as rs
import vector

from visiongraph.input.BaseDepthInput import BaseDepthInput
from visiongraph.model.types.RealSenseColorScheme import RealSenseColorScheme
from visiongraph.model.types.RealSenseFilter import RealSenseFilters
from visiongraph.util.ArgUtils import add_enum_choice_argument, add_dict_choice_argument
from visiongraph.util.MathUtils import transform_coordinates, constrain
from visiongraph.util.TimeUtils import current_millis


class RealSenseInput(BaseDepthInput):
    def __init__(self):
        super().__init__()
        self.use_infrared = False
        self.disable_emitter = False
        self.serial: Optional[str] = None

        self._exposure: Optional[float] = None
        self._gain: Optional[float] = None

        self.colorizer: Optional[rs.colorizer] = None
        self.color_scheme = RealSenseColorScheme.WhiteToBlack

        self.pipeline: Optional[rs.pipeline] = None
        self.frames: Optional[rs.composite_frame] = None
        self.align: Optional[rs.align] = None

        self.profile: Optional[rs.pipeline_profile] = None
        self.device: Optional[rs.device] = None
        self.image_sensor: Optional[rs.sensor] = None

        self._depth_frame: Optional[rs.depth_frame] = None

        # filter
        self.depth_filters: List[rs.filter] = []
        self._filters_to_enable: List[type(rs.filter)] = []

    def setup(self):
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            raise Exception("No RealSense device found!")

        self.pipeline = rs.pipeline(ctx)

        config = rs.config()

        if self.serial is not None:
            config.enable_device(serial=self.serial)

        if self.use_infrared:
            config.enable_stream(rs.stream.infrared, self.width, self.height, rs.format.y8, self.fps)
            self.align = rs.align(rs.stream.infrared)
        else:
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.align = rs.align(rs.stream.color)

        if self.enable_depth:
            self.colorizer = rs.colorizer(color_scheme=self.color_scheme.value)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            [self.depth_filters.append(f()) for f in self._filters_to_enable]

        self.profile = self.pipeline.start(config)
        self.device = self.profile.get_device()

        # set emitter
        depth_sensor = self.device.first_depth_sensor()
        if self.disable_emitter:
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        else:
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

        # setting options
        self.image_sensor = self.device.first_depth_sensor() if self.use_infrared else self.device.first_color_sensor()
        self.set_option(rs.option.enable_auto_exposure, int(not bool(self._exposure)))

        if self._exposure:
            self.set_option(rs.option.exposure, float(self._exposure))

        if self._gain:
            self.set_option(rs.option.gain, float(self._gain))

    def release(self):
        self.pipeline.stop()

    def read(self) -> (int, Optional[np.ndarray]):
        self.frames = self.pipeline.wait_for_frames()
        time_stamp = current_millis()

        if self.align:
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
        if self._depth_frame is None:
            raise Exception("Depth is not enabled for RealSense input.")

        return self._depth_frame

    def _calculate_depth_coordinates(self, x: float, y: float, depth_frame: rs.depth_frame) -> Tuple[int, int]:
        x, y = transform_coordinates(x, y, self.rotate, self.flip)

        ix = round(constrain(depth_frame.width * x, upper=depth_frame.width - 1))
        iy = round(constrain(depth_frame.height * y, upper=depth_frame.height - 1))

        return ix, iy

    def distance(self, x: float, y: float) -> float:
        depth_frame = self.depth_frame
        ix, iy = self._calculate_depth_coordinates(x, y, self.depth_frame)

        return depth_frame.get_distance(ix, iy)

    def pixel_to_point(self, x: float, y: float) -> vector.Vector3D:
        depth_frame: rs.depth_frame = self.depth_frame
        ix, iy = self._calculate_depth_coordinates(x, y, self.depth_frame)

        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        distance = depth_frame.get_distance(ix, iy)

        point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [ix, iy], distance)
        return vector.obj(x=point[0], y=point[1], z=point[2])

    @property
    def depth_map(self) -> np.ndarray:
        depth_frame = self.depth_frame
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        ts, transformed_depth = self._post_process(0, depth_colormap)
        return transformed_depth

    def configure(self, args: Namespace):
        super().configure(args)

        self.use_infrared = args.infrared

        self._exposure = args.exposure
        self._gain = args.gain
        self.serial = args.rs_serial

        self.disable_emitter = args.disable_emitter
        self.color_scheme = args.color_scheme

        # filter enabler
        if args.rs_filter is not None:
            self._filters_to_enable = args.rs_filter

    def get_option(self, option: rs.option) -> float:
        return self.image_sensor.get_option(option)

    def set_option(self, option: rs.option, value: float):
        if self.image_sensor.supports(option):
            self.image_sensor.set_option(option, value)
        else:
            logging.warning("the option {option} is not supported!")

    @property
    def gain(self) -> int:
        return int(self.get_option(rs.option.gain))

    @gain.setter
    def gain(self, value: int):
        self.set_option(rs.option.gain, value)

    @property
    def exposure(self) -> int:
        return int(self.get_option(rs.option.exposure))

    @exposure.setter
    def exposure(self, value: int):
        self.set_option(rs.option.exposure, value)

    @property
    def enable_auto_exposure(self) -> bool:
        return bool(self.get_option(rs.option.enable_auto_exposure))

    @enable_auto_exposure.setter
    def enable_auto_exposure(self, value: bool):
        self.set_option(rs.option.enable_auto_exposure, value)

    @property
    def enable_auto_white_balance(self) -> bool:
        return bool(self.get_option(rs.option.enable_auto_white_balance))

    @enable_auto_white_balance.setter
    def enable_auto_white_balance(self, value: bool):
        self.set_option(rs.option.enable_auto_white_balance, value)

    @property
    def white_balance(self) -> int:
        return int(self.get_option(rs.option.white_balance))

    @white_balance.setter
    def white_balance(self, value: int):
        value = value // 100 * 100
        self.set_option(rs.option.white_balance, value)

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(RealSenseInput, RealSenseInput).add_params(parser)
        parser.add_argument("-ir", "--infrared", action="store_true",
                            help="Use infrared as input stream (RealSense).")
        parser.add_argument("--exposure", default=None, type=float,
                            help="Exposure value (usec) for realsense input (disables auto-exposure).")
        parser.add_argument("--gain", default=None, type=float,
                            help="Gain value for realsense input (disables auto-exposure).")
        parser.add_argument("--rs-serial", default=None, type=str,
                            help="RealSense serial number to choose specific device.")
        parser.add_argument("--disable-emitter", action="store_true",
                            help="Disable RealSense IR emitter.")
        parser.add_argument("--depth", action="store_true",
                            help="Enable RealSense depth stream.")
        add_dict_choice_argument(parser, RealSenseFilters, "--rs-filter", help="RealSense depth filter",
                                 default=None, nargs="+")
        parser.add_argument("--depth-as-input", action="store_true",
                            help="Use colored depth stream as input stream.")
        add_enum_choice_argument(parser, RealSenseColorScheme, "--color-scheme",
                                 default=RealSenseColorScheme.WhiteToBlack,
                                 help="Color scheme for depth map")
