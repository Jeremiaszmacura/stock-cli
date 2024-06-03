"""Module designed to init and configure logger."""
import os
import logging
import datetime
import pathlib


def create_logger(logs_dir: str) -> tuple:
    os.makedirs(logs_dir, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_filename = f"{time_stamp}-stock-app.log"
    logs_file_patch = os.path.join(logs_dir, logs_filename)

    logging.basicConfig(filename=logs_file_patch, level=logging.INFO)

    logger = logging.getLogger(__name__)

    return logger, logs_file_patch


def get_logs_dir() -> str:
    current_dir_name = pathlib.Path(__file__).parent
    module_src_path, _ = os.path.split(current_dir_name)
    module_main_path, _ = os.path.split(module_src_path)
    logs_dir_name = "logs"
    logs_dir = os.path.join(module_main_path, logs_dir_name)
    return logs_dir


logs_dir = get_logs_dir()
logger, logs_file_patch = create_logger(logs_dir)
logging.info(f"Logger configured. Logs path: {logs_file_patch}")
