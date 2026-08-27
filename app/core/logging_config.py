import logging
import sys
from functools import lru_cache

from app.core.config import settings


@lru_cache
def get_logger(name: str = "ragger") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
    logger.propagate = False
    return logger