from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, ArgumentError
from typing import Optional

import numpy as np

from visiongraph.input.BaseInput import BaseInput
from visiongraph.model.CameraIntrinsics import CameraIntrinsics
from visiongraph.model.CameraStreamType import CameraStreamType


class BaseCamera(BaseInput, ABC):
    """
    Abstract base class for camera input.

    This class defines the interface and common functionality for camera
    input systems, including parameters for exposure, gain, and white
    balance settings.
    """

    def __init__(self):
        """
        Initializes the BaseCamera object.

        Sets initial settings for exposure, gain, and white balance to None.
        """
        super().__init__()

        self._initial_exposure: Optional[float] = None
        self._initial_gain: Optional[float] = None
        self._initial_white_balance: Optional[float] = None

    def _apply_initial_settings(self):
        """
        Applies the initial camera settings for exposure, gain, and white balance.

        Adjusts camera settings based on initial values, enabling or
        disabling auto-exposure and auto-white balance if necessary.
        """
        self.enable_auto_exposure = not bool(self._initial_exposure)
        self.enable_auto_white_balance = not bool(self._initial_white_balance)

        if self._initial_exposure:
            self.exposure = self._initial_exposure

        if self._initial_gain:
            self.gain = self._initial_gain

        if self._initial_white_balance:
            self.white_balance = self._initial_white_balance

    def configure(self, args: Namespace):
        """
        Configures the camera settings based on command line arguments.

        :param args: Command line arguments parsed into a Namespace.
        """
        super().configure(args)

        self._initial_exposure = args.exposure
        self._initial_gain = args.gain
        self._initial_white_balance = args.white_balance

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command line parameters specific to the camera settings.

        :param parser: The argument parser to add parameters to.
        """
        super(BaseCamera, BaseCamera).add_params(parser)

        try:
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
        """
        Retrieves the current gain setting of the camera.

        :return: The current gain value.
        """
        pass

    @gain.setter
    @abstractmethod
    def gain(self, value: int):
        """
        Sets the camera gain.

        :param value: The gain value to set.
        """
        pass

    @property
    @abstractmethod
    def exposure(self) -> int:
        """
        Retrieves the current exposure setting of the camera.

        :return: The current exposure value.
        """
        pass

    @exposure.setter
    @abstractmethod
    def exposure(self, value: int):
        """
        Sets the camera exposure.

        :param value: The exposure value to set.
        """
        pass

    @property
    @abstractmethod
    def enable_auto_exposure(self) -> bool:
        """
        Indicates whether auto-exposure is enabled.

        :return: True if auto-exposure is enabled, False otherwise.
        """
        pass

    @enable_auto_exposure.setter
    @abstractmethod
    def enable_auto_exposure(self, value: bool):
        """
        Enables or disables auto-exposure.

        :param value: True to enable auto-exposure, False to disable.
        """
        pass

    @property
    @abstractmethod
    def enable_auto_white_balance(self) -> bool:
        """
        Indicates whether auto-white balance is enabled.

        :return: True if auto-white balance is enabled, False otherwise.
        """
        pass

    @enable_auto_white_balance.setter
    @abstractmethod
    def enable_auto_white_balance(self, value: bool):
        """
        Enables or disables auto-white balance.

        :param value: True to enable auto-white balance, False to disable.
        """
        pass

    @property
    @abstractmethod
    def white_balance(self) -> int:
        """
        Retrieves the current white balance setting of the camera.

        :return: The current white balance value.
        """
        pass

    @white_balance.setter
    @abstractmethod
    def white_balance(self, value: int):
        """
        Sets the camera white balance.

        :param value: The white balance value to set.
        """
        pass

    @property
    def camera_matrix(self) -> np.ndarray:
        """
        Retrieves the camera matrix.

        :return: The camera matrix.
        """
        return self.get_camera_matrix()

    @property
    def fisheye_distortion(self) -> np.ndarray:
        """
        Retrieves the fisheye distortion parameters.

        :return: The fisheye distortion coefficients.
        """
        return self.fisheye_distortion

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """
        Retrieves the camera intrinsics.

        :return: The camera intrinsics object.
        """
        return self.get_intrinsics()

    @abstractmethod
    def get_camera_matrix(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        """
        Retrieves the camera matrix for a specified stream type.

        :param stream_type: The type of camera stream (default is Color).

        :return: The camera matrix.
        """
        pass

    @abstractmethod
    def get_fisheye_distortion(self, stream_type: CameraStreamType = CameraStreamType.Color) -> np.ndarray:
        """
        Retrieves the fisheye distortion coefficients for a specified stream type.

        :param stream_type: The type of camera stream (default is Color).

        :return: The fisheye distortion coefficients.
        """
        pass

    def get_intrinsics(self, stream_type: CameraStreamType = CameraStreamType.Color) -> CameraIntrinsics:
        """
        Retrieves the camera intrinsics for a specified stream type.

        :param stream_type: The type of camera stream (default is Color).

        :return: The camera intrinsics object.
        """
        return CameraIntrinsics(self.get_camera_matrix(stream_type), self.get_fisheye_distortion(stream_type))

    @property
    @abstractmethod
    def serial(self) -> str:
        """
        Retrieves the serial number of the camera.

        :return: The serial number of the camera.
        """
        pass
