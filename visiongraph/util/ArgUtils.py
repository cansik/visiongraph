import argparse
from typing import Dict, Any, Optional, Callable, Union

from visiongraph.GraphNode import GraphNode


def dict_choice(table):
    """
    Create a checker function for argparse that ensures the provided key exists in the given dictionary.

    Args:
        table (dict): A dictionary of valid choices.

    Returns:
        Callable: A function that checks if a key is valid in the dictionary.
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
    """
    Add an argument to the ArgumentParser that uses a dictionary of choices.

    Args:
        parser (argparse.ArgumentParser): The ArgumentParser to add the argument to.
        source (dict): A mapping of choice names to their corresponding values.
        name (str): The name of the argument.
        help (str, optional): A help message for the argument.
        default (Optional[Union[int, str]], optional): The default value for the argument.
        nargs (Optional[Union[str, int]], optional): The number of arguments expected.

    Returns:
        None
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

    Args:
        parser (argparse.ArgumentParser): The ArgumentParser to add the argument to.
        steps (dict): A mapping of step names to GraphNode instances.
        name (str): The name of the argument.
        help (str, optional): A help message for the argument.
        default (Optional[Union[int, str]], optional): The default value for the argument.
        add_params (bool, optional): Whether to add parameters for the GraphNode.

    Returns:
        None
    """
    add_dict_choice_argument(parser, steps, name, help, default)

    if add_params:
        for item in steps.keys():
            steps[item].add_params(parser)


def add_enum_choice_argument(parser: argparse.ArgumentParser, enum_type: Any, name: str, help: str = "",
                             default: Optional[Any] = None):
    """
    Add an argument to the ArgumentParser that uses an enumeration type for choices.

    Args:
        parser (argparse.ArgumentParser): The ArgumentParser to add the argument to.
        enum_type (enum): An enumeration type that provides valid choices.
        name (str): The name of the argument.
        help (str, optional): A help message for the argument.
        default (Optional[Any], optional): The default value for the argument.

    Returns:
        None
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

    Args:
        pipeline_node (GraphNode): The GraphNode associated with the pipeline.
        method (Callable): The method to be called on the pipeline node.
        params (Any): Additional parameters to pass to the method.
    """

    def __init__(self, pipeline_node: GraphNode, method: Callable, *params: Any):
        """
        Initializes the PipelineNodeFactory with a specific GraphNode and method.

        Args:
            pipeline_node (GraphNode): The GraphNode associated with the pipeline.
            method (Callable): The method to be called on the pipeline node.
            params (Any): Additional parameters to pass to the method.
        """
        self.pipeline_node = pipeline_node
        self.method = method
        self.params = params
