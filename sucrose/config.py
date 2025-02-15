
import os
import json
from typing import Optional

from .header import auto_get_project
from .header import ProjectHeader as Proj
from .sucrose_logger import logger


def read_config(project: Optional[Proj] = None, /):
    log_dir = auto_get_project(project).LOGS_DIR
    log_file = os.path.join(log_dir, "config.json")

    with open(log_file, 'r') as f:
        config_data = json.load(f)

    logger.info(f"config file loaded from {log_file}")

    return config_data
