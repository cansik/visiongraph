import argparse
from typing import Dict, Any, Optional, Callable, Union

from visiongraph.GraphNode import GraphNode


def dict_choice(table):
    def dict_choice_checker(key):
        try:
            item = table[key]
        except KeyError:
            choices = ", ".join(list(table.keys()))
            raise argparse.ArgumentTypeError(f"Option {key} is not defined in ({choices})")

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
                             name: str, help: str = "", default: Optional[Union[int, str]] = 0,
                             nargs: Optional[Union[str, int]] = None):
    items = list(source.keys())
    help_text = f"{help}"

    default_item = None
    if default is not None:
        if type(default) is str:
            default = items.index(default)

        default_name = items[default]
        default_item = source[items[default]]
        help_text += f", default: {default_name}."
    else:
        help_text += "."

    choices = ",".join(list(source.keys()))
    parser.add_argument(name, default=default_item, metavar=choices, nargs=nargs,
                        type=dict_choice(source), help=help_text)


def add_step_choice_argument(parser: argparse.ArgumentParser, steps: Dict[str, GraphNode],
                             name: str, help: str = "", default: Optional[Union[int, str]] = 0,
                             add_params: bool = True):
    add_dict_choice_argument(parser, steps, name, help, default)

    if add_params:
        for item in steps.keys():
            steps[item].add_params(parser)


def add_enum_choice_argument(parser: argparse.ArgumentParser, enum_type: Any, name: str, help: str = "",
                             default: Optional[Any] = None):
    values = list(enum_type)
    items = {item.name: item for item in list(enum_type)}

    if default is not None:
        default_index = values.index(default)
    else:
        default_index = 0

    add_dict_choice_argument(parser, items, name, help, default_index)


class PipelineNodeFactory:
    def __init__(self, pipeline_node: GraphNode, method: Callable, *params: Any):
        self.pipeline_node = pipeline_node
        self.method = method
        self.params = params
