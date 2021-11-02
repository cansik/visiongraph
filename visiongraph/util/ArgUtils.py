import argparse
from typing import Dict, Any, Optional

from visiongraph.PipelineStep import PipelineStep


def dict_choice(steps):
    def dict_choice_checker(arg):
        try:
            step = steps[arg]
        except ValueError:
            raise argparse.ArgumentTypeError(f"step {arg} is not defined")

        return step()

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

    default_item = None
    if default is not None:
        default_item = source[items[default]]

    parser.add_argument(name, default=default_item, choices=items, type=dict_choice(source),
                        help=f"{help}, default: {default_item}")


def add_step_choice_argument(parser: argparse.ArgumentParser, steps: Dict[str, PipelineStep],
                             name: str, help: str = "", default: Optional[int] = 0, add_params: bool = True):
    add_dict_choice_argument(parser, steps, name, help, default)

    if add_params:
        for item in steps.keys():
            steps[item].add_params(parser)
