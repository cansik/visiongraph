import argparse
from typing import Dict, Any, Optional, Union

from visiongraph.PipelineStep import PipelineStep


def dict_choice(table):
    def dict_choice_checker(key):
        try:
            item = table[key]
        except ValueError:
            raise argparse.ArgumentTypeError(f"key {key} is not defined")

        return item

    return dict_choice_checker


def float_range(mini, maxi):
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi) + "]")
        return f

    # Return function handle to checking function
    return float_range_checker


def add_dict_choice_argument(parser: argparse.ArgumentParser, source: Dict[str, Any],
                             name: str, help: str = "", default: Optional[int] = 0):
    items = list(source.keys())
    help_text = f"{help}"

    default_item = None
    if default is not None:
        default_name = items[default]
        default_item = source[items[default]]
        help_text += f", default: {default_name}."
    else:
        help_text += "."

    choices = ",".join(list(source.keys()))
    parser.add_argument(name, default=default_item, metavar=choices, type=dict_choice(source),
                        help=help_text)


def add_step_choice_argument(parser: argparse.ArgumentParser, steps: Dict[str, PipelineStep],
                             name: str, help: str = "", default: Optional[int] = 0, add_params: bool = True):
    add_dict_choice_argument(parser, steps, name, help, default)

    if add_params:
        for item in steps.keys():
            steps[item].add_params(parser)
