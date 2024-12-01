import logging


def add_logging_parameter(parser):
    """
    Adds a command-line argument for specifying the logging level.

    :param parser: The argument parser to which the logging level argument will be added.
    """
    parser.add_argument('--loglevel', default='warning',
                        choices=["critical", "error", "warning", "info", "debug"],
                        help='Provide logging level. Example --loglevel debug, default=warning')


def setup_logging(loglevel: str):
    """
    Configures the logging settings based on the specified log level.

    :param loglevel: The logging level to set, must be one of 'critical', 'error', 'warning', 'info', or 'debug'.
    """
    logging.basicConfig(level=loglevel.upper(),
                        format="%(levelname)s - %(asctime)s.%(msecs)03d - %(module)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True)
