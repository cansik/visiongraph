import logging


def add_logging_parameter(parser):
    parser.add_argument('--loglevel', default='warning',
                        choices=["critical", "error", "warning", "info", "debug"],
                        help='Provide logging level. Example --loglevel debug, default=warning')


def setup_logging(loglevel: str):
    logging.basicConfig(level=loglevel.upper(),
                        format="%(levelname)s - %(asctime)s.%(msecs)03d - %(module)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", force=True)
