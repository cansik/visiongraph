import logging
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
import pyrealsense2 as rs

from visiongraph.input.BaseInput import BaseInput
from visiongraph.util.TimeUtils import current_millis


class RealSenseInput(BaseInput):
    def __init__(self):
        super().__init__()
        self.use_infrared = False
        self.enable_depth = False

        # todo: remove duplicate values
        self._exposure: Optional[float] = None
        self._gain: Optional[float] = None

        self.pipeline: Optional[rs.pipeline] = None
        self.frames: Optional[rs.composite_frame] = None
        self.align: Optional[rs.align] = None

        self.profile: Optional[rs.pipeline_profile] = None
        self.device: Optional[rs.device] = None
        self.image_sensor: Optional[rs.sensor] = None

    def setup(self):
        #  todo: implement starting by device serial-number
        self.pipeline = rs.pipeline()

        config = rs.config()
        if self.use_infrared:
            config.enable_stream(rs.stream.infrared, self.width, self.height, rs.format.y8, self.fps)
            self.align = rs.align(rs.stream.infrared)
        else:
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.align = rs.align(rs.stream.color)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(config)

        self.device = self.profile.get_device()

        # disable emitter
        depth_sensor = self.device.first_depth_sensor()
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

        # setting options
        self.image_sensor = self.device.first_depth_sensor() if self.use_infrared else self.device.first_color_sensor()
        self.image_sensor.set_option(rs.option.enable_auto_exposure, int(not bool(self._exposure)))

        if self._exposure:
            self.image_sensor.set_option(rs.option.exposure, float(self._exposure))

        if self._gain:
            self.image_sensor.set_option(rs.option.gain, float(self._gain))

    def release(self):
        self.pipeline.stop()

    def read(self) -> (int, Optional[np.ndarray]):
        self.frames = self.pipeline.wait_for_frames()
        time_stamp = current_millis()

        if self.align:
            # alignment only happens if depth is enabled!
            self.frames = self.align.process(self.frames)

        if self.use_infrared:
            image = self.frames.get_infrared_frame()
        else:
            image = self.frames.get_color_frame()

        if not image:
            logging.warning(f"{self.__class__.__name__} could not read frame.")
            return self._post_process(time_stamp, None)

        return self._post_process(time_stamp, np.asanyarray(image.get_data()))

    def configure(self, args: Namespace):
        super().configure(args)

        self.use_infrared = args.infrared
        self._exposure = args.exposure
        self._gain = args.gain

        # todo: implement depth as input again
        # self.enable_depth = args.depth and args.depth_estimator == "realsense"

    def get_option(self, option: rs.option) -> float:
        return self.image_sensor.get_option(option)

    def set_option(self, option: rs.option, value: float):
        self.image_sensor.set_option(option, value)

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
