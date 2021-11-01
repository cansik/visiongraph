import argparse
from typing import Dict

from visiongraph.PipelineStep import PipelineStep


def step_choice(steps):
    def step_choice_checker(arg):
        try:
            step = steps[arg]
        except ValueError:
            raise argparse.ArgumentTypeError(f"step {arg} is not defined")

        return step()

    return step_choice_checker


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


def add_step_choice_argument(parser: argparse.ArgumentParser, steps: Dict[str, PipelineStep],
                             name: str, help: str = "", default: int = 0, add_params: bool = True):
    items = list(steps.keys())
    parser.add_argument(name, default=steps[items[default]], choices=items, type=step_choice(steps),
                        help=f"{help}, default: {items[default]}")

    if add_params:
        for item in items:
            steps[item].add_params(parser)
