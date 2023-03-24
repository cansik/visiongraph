import argparse


def add_source_argument(parser: argparse.ArgumentParser):
    try:
        parser.add_argument("-src", "--source", type=str, help="Generic input source for all inputs.")
    except argparse.ArgumentError as ex:
        if ex.message.startswith("conflicting"):
            return
        raise ex
