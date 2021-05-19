from pathlib import Path
from logging import getLogger, FileHandler, StreamHandler, Formatter
from logging import DEBUG, INFO, WARN, ERROR
from sys import stdout

__all__ = ['get_logger']

def get_logger(name: str, level=DEBUG):
    logs = Path('logs/')

    if not logs.exists():
        print("Created 'logs' dir")
        logs.mkdir()

    if '.py' in name:
        name = name[:-3]

    LOGGER = getLogger(name)
    LOGGER.setLevel(level)

    FORMATTER = Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    FH = FileHandler(logs.joinpath(f"{name}.log"))
    SH = StreamHandler(stdout)

    FH.setLevel(level)
    SH.setLevel(level)

    FH.setFormatter(FORMATTER)
    SH.setFormatter(FORMATTER)

    LOGGER.addHandler(FH)
    LOGGER.addHandler(SH)
    return LOGGER