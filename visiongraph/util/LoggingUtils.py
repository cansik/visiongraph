import logging


def add_logging_parameter(parser):
    parser.add_argument('--loglevel', default='warning',
                        choices=["critical", "error", "warning", "info", "debug"],
                        help='Provide logging level. Example --loglevel debug, default=warning')


def setup_logging(loglevel):
    logging.basicConfig(level=loglevel.upper(),
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
