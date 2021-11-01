import argparse

from visiongraph.input import InputProviders, add_input_step_choices
from visiongraph.util.ArgUtils import add_step_choice_argument

parser = argparse.ArgumentParser("visiongraph", description="just an example help text")
# add_step_choice_argument(parser, InputProviders, "--input", help="Image input provider")
add_input_step_choices(parser)

args = parser.parse_args()

print(args.input)
