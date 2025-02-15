
from typing import Optional, Dict, Any

from .const import *
from .header import (
    ProjectHeader,
    get_current_project,
    save_ckpts,
    load_ckpts,
)
from .logger import start_pytorch_tensorboard
from .config import read_config


def start_project(
        work_dir: str, name: str, *,
        epoch_prefix: str = 'e',
        backend: str = 'pytorch',
        ckpts_ext: Optional[str] = None,
        save_extra_kwds: Dict[str, Any] = {},
        load_extra_kwds: Dict[str, Any] = {}):
    return ProjectHeader(work_dir, name)

def step():
    get_current_project().step()
