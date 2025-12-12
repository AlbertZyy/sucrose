
__all__ = ["start_pytorch_tensorboard_impl"]

from ..sucrose_logger import logger


def start_pytorch_tensorboard_impl(logdir: str, **kwargs):
    from torch.utils.tensorboard import SummaryWriter
    writter = SummaryWriter(logdir, **kwargs)

    return writter
