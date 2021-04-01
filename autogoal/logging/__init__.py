import logging
import warnings

from rich.logging import RichHandler
from rich.console import Console


_CONSOLE = Console()


def setup(level="INFO"):
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=_CONSOLE)],
    )
    logging.captureWarnings(True)


def logger() -> logging.Logger:
    return logging.getLogger("autogoal")


def console() -> Console:
    return _CONSOLE
