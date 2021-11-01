import argparse

from visiongraph.input import InputProviders
from visiongraph.util.ArgUtils import add_step_choice_argument

parser = argparse.ArgumentParser("visiongraph", description="just an example help text")
add_step_choice_argument(parser, InputProviders, "--input", help="Image input provider")

args = parser.parse_args()

print(args.input)
