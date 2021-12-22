import argparse
from argparse import _ArgumentGroup
from typing import Union

from visiongraph.input.AzureKinectInput import AzureKinectInput
from visiongraph.input.RealSenseInput import RealSenseInput
from visiongraph.input.VideoCaptureInput import VideoCaptureInput
from visiongraph.util.ArgUtils import add_step_choice_argument


InputProviders = {
    "video-capture": VideoCaptureInput,
    "realsense": RealSenseInput,
    "azure": AzureKinectInput,
}


def add_input_step_choices(parser: Union[argparse.ArgumentParser, _ArgumentGroup], default: int = 0,
                           add_params: bool = True):
    add_step_choice_argument(parser, InputProviders, "--input", help="Image input provider",
                             default=default, add_params=add_params)
