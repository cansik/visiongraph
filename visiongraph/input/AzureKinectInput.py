import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, Tuple

import cv2
import numpy as np
import pyk4a
from pyk4a import PyK4A, PyK4ACapture, Config, PyK4ARecord, PyK4APlayback, ImageFormat, CalibrationType

from visiongraph.input.BaseDepthCamera import BaseDepthCamera
from visiongraph.util.ArgUtils import add_enum_choice_argument
from visiongraph.util.CollectionUtils import default_value_dict
from visiongraph.util.MathUtils import transform_coordinates, constrain
from visiongraph.util.TimeUtils import current_millis


class AzureKinectInput(BaseDepthCamera):
    _HeightToResolutionMapping = default_value_dict(pyk4a.ColorResolution.RES_720P,
                                                    {
                                                        720: pyk4a.ColorResolution.RES_720P,
                                                        1080: pyk4a.ColorResolution.RES_1080P,
                                                        1440: pyk4a.ColorResolution.RES_1440P,
                                                        1536: pyk4a.ColorResolution.RES_1536P,
                                                        2160: pyk4a.ColorResolution.RES_2160P,
                                                        3072: pyk4a.ColorResolution.RES_3072P,
                                                    })

    _FPSToK4AFPSMapping = default_value_dict(pyk4a.FPS.FPS_30,
                                             {
                                                 5: pyk4a.FPS.FPS_5,
                                                 15: pyk4a.FPS.FPS_15,
                                                 30: pyk4a.FPS.FPS_30,
                                             })

    def __init__(self, device_id: int = 0):
        super().__init__()
        self.sync_frames: bool = True
        self.align_frames: bool = False

        self.min_clipping: Optional[int] = 0
        self.max_clipping: Optional[int] = 5000
        self.color_map: Optional[int] = cv2.COLORMAP_JET

        self.device: Optional[PyK4A] = None
        self.capture: Optional[PyK4ACapture] = None

        self.device_id: int = device_id
        self.color_resolution: Optional[pyk4a.ColorResolution] = None
        self.color_format: pyk4a.ImageFormat = pyk4a.ImageFormat.COLOR_BGRA32
        self.depth_mode: pyk4a.DepthMode = pyk4a.DepthMode.NFOV_UNBINNED

        self.config: Optional[Config] = None

        # recording / playback
        self.input_mkv_file: Optional[str] = None
        self.output_mkv_file: Optional[str] = None

        self._record: Optional[PyK4ARecord] = None
        self._playback: Optional[PyK4APlayback] = None

    def setup(self, config: Optional[Config] = None):
        if self.input_mkv_file is not None:
            logging.info(f"Playing mkv file from {self.input_mkv_file}")
            self._playback = PyK4APlayback(self.input_mkv_file)
            self._playback.open()
            self.color_format = self._playback.configuration["color_format"]
            return

        if self.device_count == 0:
            raise Exception("No Azure Kinect device found!")

        if config is not None:
            self.device = PyK4A(config=config, device_id=self.device_id)
            self.device.start()
        else:
            config = Config()

            if self.color_resolution is None:
                config.color_resolution = AzureKinectInput._HeightToResolutionMapping[self.height]
            else:
                config.color_resolution = self.color_resolution

            config.color_format = self.color_format
            config.camera_fps = AzureKinectInput._FPSToK4AFPSMapping[self.fps]
            config.depth_mode = pyk4a.DepthMode.OFF
            config.synchronized_images_only = False

            if self.use_infrared:
                config.depth_mode = pyk4a.DepthMode.PASSIVE_IR
                config.synchronized_images_only = self.sync_frames

            if self.enable_depth:
                config.depth_mode = self.depth_mode
                config.synchronized_images_only = self.sync_frames

            self.config = config
            self.device = PyK4A(config=config, device_id=self.device_id)
            self.device.start()

        # set options
        self._apply_initial_settings()

        # recording
        if self.output_mkv_file is not None:
            logging.info(f"Starting recording to {self.output_mkv_file}")
            self._record = PyK4ARecord(device=self.device, config=config, path=self.output_mkv_file)
            self._record.create()

    def read(self) -> (int, Optional[np.ndarray]):
        self._read_next_capture()
        time_stamp = current_millis()

        if self._record is not None:
            self._record.write_capture(self.capture)

        if self.enable_depth and self.use_depth_as_input:
            depth = self.capture.depth
            image = self._colorize(depth, (self.min_clipping, self.max_clipping), self.color_map)
        else:
            if self.use_infrared:
                ir_frame = self.capture.transformed_ir if self.align_frames else self.capture.ir
                image = self._colorize(ir_frame, (None, None), None)
            else:
                image = self.capture.transformed_color if self.align_frames else self.capture.color
                if image is not None:
                    image = self._convert_to_bgra_if_required(self.color_format, image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if image is None:
            logging.warning("could not read frame.")
            return self._post_process(time_stamp, None)

        return self._post_process(time_stamp, image)

    def release(self):
        if self._record is not None:
            self._record.flush()
            self._record.close()
            logging.info(f"Recording has been written to {self.output_mkv_file}")

        if self._playback is not None:
            self._playback.close()
        else:
            self.device.stop()

    def distance(self, x: float, y: float) -> float:
        depth_frame = self.capture.depth
        h, w = depth_frame.shape[:2]

        x, y = transform_coordinates(x, y, self.rotate, self.flip)

        ix = round(constrain(w * x, upper=w - 1))
        iy = round(constrain(h * y, upper=h - 1))

        # convert mm into m
        return depth_frame[iy, ix] / 1000

    def _read_next_capture(self):
        if self._playback is None:
            self.capture = self.device.get_capture()
            return

        try:
            self.capture = self._playback.get_next_capture()
        except EOFError:
            self._playback.seek(0)
            self.capture = self._playback.get_next_capture()

    @staticmethod
    def _colorize(image: np.ndarray,
                  clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
                  colormap: Optional[int] = None) -> np.ndarray:
        if clipping_range[0] or clipping_range[1]:
            img = image.clip(clipping_range[0], clipping_range[1])
        else:
            img = image.copy()
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if colormap is not None:
            img = cv2.applyColorMap(img, colormap)
        return img

    @staticmethod
    def _convert_to_bgra_if_required(color_format: ImageFormat, color_image):
        if color_format == ImageFormat.COLOR_BGRA32:
            return color_image

        # examples for all possible pyk4a.ColorFormats
        if color_format == ImageFormat.COLOR_MJPG:
            color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        elif color_format == ImageFormat.COLOR_NV12:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
            # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
            # h, w = color_image.shape[0:2]
            # h = h // 3 * 2
            # luminance = color_image[:h]
            # chroma = color_image[h:, :w//2]
            # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
        elif color_format == ImageFormat.COLOR_YUY2:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
        return color_image

    @property
    def depth_map(self) -> np.ndarray:
        return self._colorize(self.capture.depth, (self.min_clipping, self.max_clipping), self.color_map)

    @property
    def depth_buffer(self) -> np.ndarray:
        return self.capture.depth

    @property
    def device_count(self) -> int:
        return pyk4a.connected_device_count()

    def configure(self, args: Namespace):
        super().configure(args)
        self.align_frames = args.k4a_align
        self.device_id = args.k4a_device

        self.output_mkv_file = args.k4a_record_mkv
        self.input_mkv_file = args.k4a_play_mkv

        self.depth_mode = args.k4a_depth_mode
        self.color_resolution = args.k4a_color_resolution
        self.color_format = args.k4a_color_format

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(AzureKinectInput, AzureKinectInput).add_params(parser)
        parser.add_argument("--k4a-align", action="store_true",
                            help="Align azure frames to depth frame.")
        parser.add_argument("--k4a-device", type=int, default=0, help="Azure device id.")

        parser.add_argument("--k4a-play-mkv", type=str, default=None,
                            help="Path to a pre-recorded bag file for playback.")
        parser.add_argument("--k4a-record-mkv", type=str, default=None,
                            help="Path to a mkv file to store the current recording.")

        add_enum_choice_argument(parser, pyk4a.DepthMode, "--k4a-depth-mode", default=pyk4a.DepthMode.NFOV_UNBINNED,
                                 help="Azure depth mode")
        add_enum_choice_argument(parser, pyk4a.ColorResolution, "--k4a-color-resolution",
                                 default=pyk4a.ColorResolution.RES_720P,
                                 help="Azure color resolution (overwrites input-size)")
        add_enum_choice_argument(parser, pyk4a.ImageFormat, "--k4a-color-format",
                                 default=pyk4a.ImageFormat.COLOR_BGRA32,
                                 help="Azure color image format")

        # todo: add more azure specific options like depth mode

    @property
    def gain(self) -> int:
        return self.device.gain

    @gain.setter
    def gain(self, value: int):
        self.device.gain = value

    @property
    def exposure(self) -> int:
        return self.device.exposure

    @exposure.setter
    def exposure(self, value: int):
        self.device.exposure = value

    @property
    def enable_auto_exposure(self) -> bool:
        return self.device.exposure_mode_auto

    @enable_auto_exposure.setter
    def enable_auto_exposure(self, value: bool):
        self.device.exposure_mode_auto = value

    @property
    def enable_auto_white_balance(self) -> bool:
        return self.device.whitebalance_mode_auto

    @enable_auto_white_balance.setter
    def enable_auto_white_balance(self, value: bool):
        self.device.whitebalance_mode_auto = value

    @property
    def white_balance(self) -> int:
        return self.device.whitebalance

    @white_balance.setter
    def white_balance(self, value: int):
        value = value // 10 * 10
        self.device.whitebalance = value

    @property
    def camera_matrix(self) -> np.ndarray:
        calibration = self.device.calibration
        return calibration.get_camera_matrix(CalibrationType.DEPTH)

    @property
    def fisheye_distortion(self) -> np.ndarray:
        calibration = self.device.calibration
        return calibration.get_distortion_coefficients(CalibrationType.DEPTH)

    @property
    def serial(self) -> str:
        return self.device.selected_serial
