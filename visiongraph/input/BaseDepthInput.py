from abc import ABC
from argparse import Namespace, ArgumentParser, ArgumentError
from typing import Optional

from visiongraph.input.BaseInput import BaseInput
from visiongraph.model.DepthBuffer import DepthBuffer


class BaseDepthInput(DepthBuffer, BaseInput, ABC):
    def __init__(self):
        super().__init__()
        self.enable_depth: bool = False
        self.use_depth_as_input: bool = False

        self.depth_width: Optional[int] = None
        self.depth_height: Optional[int] = None

    def configure(self, args: Namespace):
        super().configure(args)

        self.enable_depth = args.depth
        self.use_depth_as_input = args.depth_as_input

        if self.use_depth_as_input:
            self.enable_depth = True

    @staticmethod
    def add_params(parser: ArgumentParser):
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
