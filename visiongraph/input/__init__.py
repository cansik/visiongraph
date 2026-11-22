import argparse
import logging
import sys
from argparse import _ArgumentGroup
from functools import cache
from importlib import import_module

from visiongraph.input.BaseInput import BaseInput
from visiongraph.input.ImageInput import ImageInput
from visiongraph.input.VideoCaptureInput import VideoCaptureInput
from visiongraph.util.ArgUtils import add_step_choice_argument

InputProviders: dict[str, type[BaseInput]]

_OPTIONAL_INPUT_PROVIDERS = {
    "realsense": "RealSenseInput",
    "azure": "AzureKinectInput",
    "camgear": "CamGearInput",
    "oak1": "Oak1Input",
    "oakd": "OakDInput",
    "zed": "ZEDInput",
}
if sys.platform != "win32":
    _OPTIONAL_INPUT_PROVIDERS.pop("azure")


def _load_input_class(name: str) -> type[BaseInput]:
    provider = getattr(import_module(f".{name}", __name__), name)
    # Expose the class on the package under the same name as its module.
    globals()[name] = provider
    return provider


@cache
def _get_input_providers() -> dict[str, type[BaseInput]]:
    providers: dict[str, type[BaseInput]] = {"video-capture": VideoCaptureInput, "image": ImageInput}
    for key, name in _OPTIONAL_INPUT_PROVIDERS.items():
        try:
            providers[key] = _load_input_class(name)
        except ImportError as exc:
            logging.info(f"Input provider {key} unavailable: {exc}")
    return providers


def __getattr__(name: str) -> dict[str, type[BaseInput]] | type[BaseInput]:
    if name == "InputProviders":
        return _get_input_providers()
    if name in _OPTIONAL_INPUT_PROVIDERS.values():
        try:
            return _load_input_class(name)
        except ImportError as exc:
            raise AttributeError(name) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def add_input_step_choices(
    parser: argparse.ArgumentParser | _ArgumentGroup, default: int | str = 0, add_params: bool = True
):
    """
    Adds input step choices to the given parser.

    This function modifies the provided parser to include choices for input providers.
    It uses predefined input providers which are conditionally imported based on availability.

    :param parser: The parser or argument group to which the input step choices will be added.
    :param default: The default value for the input choice. Default is 0.
    :param add_params: A flag indicating whether to add additional parameters for the input choice. Default is True.
    """
    add_step_choice_argument(
        parser, _get_input_providers(), "--input", help="Image input provider", default=default, add_params=add_params
    )
