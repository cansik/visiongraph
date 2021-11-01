import argparse

from visiongraph.input.RealSenseInput import RealSenseInput
from visiongraph.input.VideoCaptureInput import VideoCaptureInput
from visiongraph.util.ArgUtils import add_step_choice_argument

InputProviders = {
    "video-capture": VideoCaptureInput,
    "realsense": RealSenseInput
}


def add_input_step_choices(parser: argparse.ArgumentParser, default: int = 0, add_params: bool = True):
    add_step_choice_argument(parser, InputProviders, "--input", help="Image input provider",
                             default=default, add_params=add_params)
