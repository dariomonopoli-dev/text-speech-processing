"""Logger factory shared by all pipeline scripts."""

import logging
import os
import sys

LOG_DIRECTORY = "logs"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"


def get_logger(name: str = "main") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = logging.FileHandler(os.path.join(LOG_DIRECTORY, f"{name}_logs.txt"))
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger
