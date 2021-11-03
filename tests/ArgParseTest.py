import argparse

from visiongraph.input import add_input_step_choices
from visiongraph.util.LoggingUtils import add_logging_parameter

parser = argparse.ArgumentParser("visiongraph", description="just an example help text")
add_logging_parameter(parser)
input_group = parser.add_argument_group("input provider")
add_input_step_choices(input_group)

args = parser.parse_args()

print(args.input)