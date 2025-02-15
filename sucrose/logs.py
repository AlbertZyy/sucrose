
from typing import Optional

from .header import auto_get_project
from .sucrose_logger import logger


def start_pytorch_tensorboard(project: Optional[str] = None, **kwargs):
    log_dir = auto_get_project(project).LOGS_DIR

    from torch.utils.tensorboard import SummaryWriter
    writter = SummaryWriter(log_dir, **kwargs)
    logger.info(f"SummaryWriter started at {log_dir}")
    return writter
