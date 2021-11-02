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

        self.exposure: Optional[float] = None

        self.pipeline: Optional[rs.pipeline] = None
        self.frames: Optional[rs.composite_frame] = None
        self.align: Optional[rs.align] = None

    def setup(self):
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

        profile = self.pipeline.start(config)

        device = profile.get_device()

        # disable emitter
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.emitter_enabled, 0)

        # setting options
        image_sensor = device.first_depth_sensor() if self.use_infrared else device.first_color_sensor()
        image_sensor.set_option(rs.option.enable_auto_exposure, int(not bool(self.exposure)))
        if self.exposure:
            image_sensor.set_option(rs.option.exposure, float(self.exposure))

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
            return time_stamp, None

        return time_stamp, np.asanyarray(image.get_data())

    def configure(self, args: Namespace):
        self.use_infrared = args.infrared
        self.exposure = args.exposure

        # todo: implement depth as input again
        # self.enable_depth = args.depth and args.depth_estimator == "realsense"

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(RealSenseInput, RealSenseInput).add_params(parser)
        parser.add_argument("-ir", "--infrared", action="store_true",
                            help="Use infrared as input stream (RealSense).")
        parser.add_argument("--exposure", default=None, type=float,
                            help="Exposure value (usec) for realsense input (disables auto-exposure).")
