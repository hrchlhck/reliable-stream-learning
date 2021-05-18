import logging
import sys

__all__ = ['get_logger']

def get_logger(name: str, level=logging.DEBUG):
    if '.py' in name:
        name = name[:-3]

    LOGGER = logging.getLogger(name)
    LOGGER.setLevel(level)

    FORMATTER = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s - %(message)s")
    FH = logging.FileHandler(f"{name}.log")
    SH = logging.StreamHandler(sys.stdout)

    FH.setLevel(level)
    SH.setLevel(level)

    FH.setFormatter(FORMATTER)
    SH.setFormatter(FORMATTER)

    LOGGER.addHandler(FH)
    LOGGER.addHandler(SH)
    return LOGGER