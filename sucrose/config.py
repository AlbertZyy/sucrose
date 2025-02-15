
import os
import json
from typing import Optional

from .header import auto_get_project


def read_config(project: Optional[str] = None, /):
    log_dir = auto_get_project(project).LOGS_DIR

    with open(os.path.join(log_dir, "config.json"), 'r') as f:
        config_data = json.load(f)

    return config_data
