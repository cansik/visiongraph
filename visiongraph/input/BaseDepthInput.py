from abc import ABC
from argparse import Namespace, ArgumentParser, ArgumentError
from typing import Optional

from visiongraph.input.BaseInput import BaseInput
from visiongraph.model.DepthBuffer import DepthBuffer


class BaseDepthInput(DepthBuffer, BaseInput, ABC):
    """
    Abstract base class for depth input handling, inheriting from DepthBuffer and BaseInput.
    This class manages depth input configuration and parameters
    for processing depth data from sources like cameras.
    """

    def __init__(self):
        """
        Initializes the BaseDepthInput with default settings for depth input management.
        """
        super().__init__()
        self.enable_depth: bool = False
        self.use_depth_as_input: bool = False

        self.depth_width: Optional[int] = None
        self.depth_height: Optional[int] = None

    def configure(self, args: Namespace):
        """
        Configures the depth input settings based on command line arguments.

        :param args: The namespace containing command line argument values.
        """
        super().configure(args)

        self.enable_depth = args.depth
        self.use_depth_as_input = args.depth_as_input

        if self.use_depth_as_input:
            self.enable_depth = True

    @staticmethod
    def add_params(parser: ArgumentParser):
        """
        Adds command line parameters for depth input configuration to the argument parser.

        :param parser: The argument parser to add parameters to.
        """
        super(BaseDepthInput, BaseDepthInput).add_params(parser)

        try:
            parser.add_argument("--depth", action="store_true",
                                help="Enable RealSense depth stream.")
            parser.add_argument("--depth-as-input", action="store_true",
                                help="Use colored depth stream as input stream.")
        except ArgumentError as ex:
            if ex.message.startswith("conflicting"):
                return
            raise ex
