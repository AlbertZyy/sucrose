
from typing import Optional

from .header import auto_get_project


def start_pytorch_tensorboard(project: Optional[str] = None, **kwargs):
    log_dir = auto_get_project(project).LOGS_DIR

    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir, **kwargs)
