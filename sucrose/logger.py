
from .header import get_current_project


def start_pytorch_tensorboard(**kwargs):
    log_dir = get_current_project().LOGS_DIR

    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir, **kwargs)
