import argparse


def add_source_argument(parser: argparse.ArgumentParser):
    """
    Adds a source argument to the provided argument parser.

    :param parser: The argument parser to which the source argument will be added.

    :raises argparse.ArgumentError: If there is a conflicting argument when adding the source argument.
    """
    try:
        parser.add_argument("-src", "--source", type=str, help="Generic input source for all inputs.")
    except argparse.ArgumentError as ex:
        if ex.message.startswith("conflicting"):
            return
        raise ex
