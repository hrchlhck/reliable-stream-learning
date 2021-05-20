from pathlib import Path
from logging import getLogger, FileHandler, StreamHandler, Formatter
from logging import DEBUG, INFO, WARN, ERROR
from sys import stdout
from constants import LOGS_PATH

__all__ = ['get_logger']

def get_logger(name: str, level=DEBUG):
    if not LOGS_PATH.exists():
        print("Created 'logs' dir")
        LOGS_PATH.mkdir()

    if '.py' in name:
        name = name[:-3]

    LOGGER = getLogger(name)
    LOGGER.setLevel(level)

    FORMATTER = Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    FH = FileHandler(LOGS_PATH.joinpath(f"{name}.log"))
    SH = StreamHandler(stdout)

    FH.setLevel(level)
    SH.setLevel(level)

    FH.setFormatter(FORMATTER)
    SH.setFormatter(FORMATTER)

    LOGGER.addHandler(FH)
    LOGGER.addHandler(SH)
    return LOGGER