import argparse
from typing import Dict, Any, Optional, Callable, Union

from visiongraph.GraphNode import GraphNode


def dict_choice(table):
    """
    Create a checker function for argparse that ensures the provided key exists in the given dictionary.

    :param table: A dictionary of valid choices.

    :return: A function that checks if a key is valid in the dictionary.
    """

    def dict_choice_checker(key):
        try:
            item = table[key]
        except KeyError:
            choices = ", ".join(list(table.keys()))
            raise argparse.ArgumentTypeError(f"Option {key} is not defined in ({choices})")

        return item

    return dict_choice_checker


def float_range(mini, maxi):
    """
Return function handle of an argument type function for
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument
         """

    # Define the function with default arguments
    def float_range_checker(arg):
        """
New Type function for argparse - a float within predefined range.
"""

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
    """
    Add an argument to the ArgumentParser that uses a dictionary of choices.

    :param parser: The ArgumentParser to add the argument to.
    :param source: A mapping of choice names to their corresponding values.
    :param name: The name of the argument.
    :param help: A help message for the argument.
    :param default: The default value for the argument.
    :param nargs: The number of arguments expected.

    """
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
    """
    Add an argument to the ArgumentParser that allows for choosing a step from a dictionary of GraphNodes.

    :param parser: The ArgumentParser to add the argument to.
    :param steps: A mapping of step names to GraphNode instances.
    :param name: The name of the argument.
    :param help: A help message for the argument.
    :param default: The default value for the argument.
    :param add_params: Whether to add parameters for the GraphNode.

    """
    add_dict_choice_argument(parser, steps, name, help, default)

    if add_params:
        for item in steps.keys():
            steps[item].add_params(parser)


def add_enum_choice_argument(parser: argparse.ArgumentParser, enum_type: Any, name: str, help: str = "",
                             default: Optional[Any] = None):
    """
    Add an argument to the ArgumentParser that uses an enumeration type for choices.

    :param parser: The ArgumentParser to add the argument to.
    :param enum_type: An enumeration type that provides valid choices.
    :param name: The name of the argument.
    :param help: A help message for the argument.
    :param default: The default value for the argument.

    """
    values = list(enum_type)
    items = {item.name: item for item in list(enum_type)}

    if default is not None:
        default_index = values.index(default)
    else:
        default_index = 0

    add_dict_choice_argument(parser, items, name, help, default_index)


class PipelineNodeFactory:
    """
    A factory class for creating pipeline nodes.

    :param pipeline_node: The GraphNode associated with the pipeline.
    :param method: The method to be called on the pipeline node.
    :param params: Additional parameters to pass to the method.
    """

    def __init__(self, pipeline_node: GraphNode, method: Callable, *params: Any):
        """
        Initializes the PipelineNodeFactory with a specific GraphNode and method.

        :param pipeline_node: The GraphNode associated with the pipeline.
        :param method: The method to be called on the pipeline node.
        :param params: Additional parameters to pass to the method.
        """
        self.pipeline_node = pipeline_node
        self.method = method
        self.params = params
