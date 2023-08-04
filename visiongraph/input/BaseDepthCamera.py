from abc import ABC
from argparse import ArgumentParser, Namespace, ArgumentError

from visiongraph.input.BaseCamera import BaseCamera
from visiongraph.input.BaseDepthInput import BaseDepthInput


class BaseDepthCamera(BaseCamera, BaseDepthInput, ABC):
    def __init__(self):
        super().__init__()

        self.use_infrared = False

    def configure(self, args: Namespace):
        super().configure(args)

        self.use_infrared = args.infrared

    @staticmethod
    def add_params(parser: ArgumentParser):
        super(BaseDepthCamera, BaseDepthCamera).add_params(parser)
        BaseDepthInput.add_params(parser)

        try:
            parser.add_argument("-ir", "--infrared", action="store_true",
                                help="Use infrared as input stream.")
        except ArgumentError as ex:
            if ex.message.startswith("conflicting"):
                return
            raise ex
