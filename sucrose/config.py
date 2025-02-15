
import os
import json

from .header import get_current_project


def read_config():
    log_dir = get_current_project().LOGS_DIR

    with open(os.path.join(log_dir, "config.json"), 'r') as f:
        config_data = json.load(f)

    return config_data
