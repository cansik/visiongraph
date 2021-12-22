from abc import ABC
from argparse import Namespace

from visiongraph.input.BaseInput import BaseInput
from visiongraph.model.DepthBuffer import DepthBuffer


class BaseDepthInput(DepthBuffer, BaseInput, ABC):
    def __init__(self):
        super().__init__()
        self.enable_depth = False
        self.use_depth_as_input = False

    def configure(self, args: Namespace):
        super().configure(args)

        self.enable_depth = args.depth
        self.use_depth_as_input = args.depth_as_input

        if self.use_depth_as_input:
            self.enable_depth = True
