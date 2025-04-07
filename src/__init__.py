import logging
import sys
import warnings

warnings.filterwarnings("ignore")

log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(log_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

if root_logger.hasHandlers():
    root_logger.handlers.clear()

root_logger.addHandler(log_handler)
