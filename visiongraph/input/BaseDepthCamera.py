from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, ArgumentError
from typing import Optional

import numpy as np

from visiongraph.input.BaseDepthInput import BaseDepthInput
from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.model.CameraStreamType import CameraStreamType


class BaseDepthCamera(BaseDepthInput, ABC):
    def __init__(self):
        super().__init__()

        self.use_infrared = False

        self._initial_exposure: Optional[float] = None
        self._initial_gain: Optional[float] = None
        self._initial_white_balance: Optional[float] = None

    def _apply_initial_settings(self):
        self.enable_auto_exposure = not bool(self._initial_exposure)
        self.enable_auto_white_balance = not bool(self._initial_white_balance)

        if self._initial_exposure:
            self.exposure = self._initial_exposure

        if self._initial_gain:
            self.gain = self._initial_gain

        if self._initial_white_balance:
            self.white_balance = self._initial_white_balance

    def configure(self, args: Namespace):
        super().configure(args)

        self.use_infrared = args.infrared

        self._initial_exposure = args.exposure
        self._initial_gain = args.gain
        self._initial_white_balance = args.white_balance

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(BaseDepthCamera, BaseDepthCamera).add_params(parser)

        try:
            parser.add_argument("-ir", "--infrared", action="store_true",
                                help="Use infrared as input stream.")

            parser.add_argument("--exposure", default=None, type=int,
                                help="Exposure value (usec) for depth camera input (disables auto-exposure).")
            parser.add_argument("--gain", default=None, type=int,
                                help="Gain value for depth input (disables auto-exposure).")
            parser.add_argument("--white-balance", default=None, type=int,
                                help="White-Balance value for depth input (disables auto-white-balance).")
        except ArgumentError as ex:
            if ex.message.startswith("conflicting"):
                return
            raise ex

    @property
    @abstractmethod
    def gain(self) -> int:
        pass

    @gain.setter
    @abstractmethod
    def gain(self, value: int):
        pass

    @property
    @abstractmethod
    def exposure(self) -> int:
        pass

    @exposure.setter
    @abstractmethod
    def exposure(self, value: int):
        pass

    @property
    @abstractmethod
    def enable_auto_exposure(self) -> bool:
        pass

    @enable_auto_exposure.setter
    @abstractmethod
    def enable_auto_exposure(self, value: bool):
        pass

    @property
    @abstractmethod
    def enable_auto_white_balance(self) -> bool:
        pass

    @enable_auto_white_balance.setter
    @abstractmethod
    def enable_auto_white_balance(self, value: bool):
        pass

    @property
    @abstractmethod
    def white_balance(self) -> int:
        pass

    @white_balance.setter
    @abstractmethod
    def white_balance(self, value: int):
        pass

    @property
    def camera_matrix(self) -> np.ndarray:
        return self.get_camera_matrix()

    @property
    def fisheye_distortion(self) -> np.ndarray:
        return self.fisheye_distortion

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self.get_intrinsics()

    @abstractmethod
    def get_camera_matrix(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        pass

    @abstractmethod
    def get_fisheye_distortion(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        pass

    def get_intrinsics(self, stream_type: CameraStreamType = CameraStreamType.Color) -> CameraIntrinsics:
        return CameraIntrinsics(self.get_camera_matrix(stream_type), self.get_fisheye_distortion(stream_type))

    @property
    @abstractmethod
    def serial(self) -> str:
        pass
