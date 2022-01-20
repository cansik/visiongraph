from abc import ABC
from argparse import Namespace
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
