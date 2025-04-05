import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOGGER_NAME = "biasx_experiment_logger"


def setup_logger(
    log_file: str,
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
) -> logging.Logger:

    logger = logging.getLogger(LOGGER_NAME)
    if logger.hasHandlers():
        return logger

    logger.setLevel(min(console_log_level, file_log_level))
    log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(file_log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            logger.error(f"Failed setup for file handler at {log_file}: {e}", exc_info=False)

    logger.propagate = False
    return logger
