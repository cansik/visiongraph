from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np

from visiongraph.PipelineStep import PipelineStep


class BaseInput(PipelineStep, ABC):
    @abstractmethod
    def __init__(self):
        self.width = 640
        self.height = 480
        self.fps = 30

    @abstractmethod
    def read(self) -> (int, Optional[np.ndarray]):
        pass

    @abstractmethod
    def configure(self, args: Namespace):
        self.width, self.height = args.input_size
        self.fps = args.input_fps

    @staticmethod
    @abstractmethod
    def add_params(parser: ArgumentParser):
        try:
            parser.add_argument("--input-size", default=[640, 480], type=int, nargs=2, metavar=('width', 'height'),
                                help="Requested input media size.")
            parser.add_argument("--input-fps", default=30, type=int, help="Requested input media framerate.")
        except:
            pass
