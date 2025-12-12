
__all__ = [
    "find_latest_epoch",
    "Scenario",
    "get_current_scenario",
    "auto_get_scenario"
]

import os, re, yaml
import threading
from typing import Any, TypeVar
from collections.abc import Callable

from ..sucrose_logger import logger
from ..config import *
from .ckpt import *
from .logs import *

_R = TypeVar("_R")


LOCAL_THREAD = threading.local()


def set_current(key: str, data):
    setattr(LOCAL_THREAD, key, data)


def get_current(key: str):
    if hasattr(LOCAL_THREAD, key):
        return getattr(LOCAL_THREAD, key)
    else:
        return None


def find_latest_epoch(ckpts_dir: str, filename_pattern: str) -> int:
    """Look into the checkpoint directory and find the latest epoch.
    Return `0` if no file found."""
    if not os.path.exists(ckpts_dir):
        return 0

    all_ckpts = os.listdir(ckpts_dir)
    max_epoch = 0

    for name in all_ckpts:
        res = re.match(filename_pattern, name)
        if res is None:
            continue
        else:
            epoch = int(res.group(1))
            if epoch > max_epoch:
                max_epoch = epoch

    return max_epoch


def load_config(work_dir: str) -> dict[str, Any]:
    log_file = os.path.join(work_dir, "config.yaml")

    if not os.path.exists(log_file):
        raise FileNotFoundError(
            f"config.yaml not found in the work directory: {work_dir}"
        )

    with open(log_file, 'r') as f:
        config_data = yaml.safe_load(f)

    assert isinstance(config_data, dict), "config.yaml is expected to be a dict"
    logger.info(f"config file loaded from {log_file}")

    return config_data


class Scenario():
    """Provide scenarios to manage file paths and names."""
    def __init__(
        self,
        work_dir: str,
        name: str,
        *,
        meta_domain: str = "workspace"
    ):
        self.NAME = name
        self.WORK_DIR = work_dir
        self.CONFIG = load_config(work_dir)

        context = {"data": self.CONFIG, "domain": meta_domain}

        self.CKPTS_FOLDER = lookup(**context, field="ckpts_folder")
        self.LOGS_FOLDER  = lookup(**context, field="logs_folder")
        self.EPOCH_PREFIX = lookup(**context, field="epoch_prefix")
        self.CKPTS_EXT    = lookup(**context, field="ckpts_extension").lstrip('.')
        self.STEP_KEY     = lookup(**context, field="step_key")

        self.LAST_EPOCH = find_latest_epoch(
            self.CKPTS_DIR,
            f'{name.replace("/", "_")}_{self.EPOCH_PREFIX}([0-9]*).{self.CKPTS_EXT}'
        )
        self._step = 0 # number of steps finished, index of the next
        self._local_epoch = 0


    def __del__(self):
        if self._local_epoch != 0:
            logger.warning(f"There are still {self._local_epoch} epochs that "
                           "are not saved as checkpoint files by `save_state_dict()`. ")

    @property
    def CKPTS_DIR(self):
        return os.path.join(self.WORK_DIR, self.CKPTS_FOLDER, self.NAME)

    @property
    def LOGS_DIR(self):
        return os.path.join(self.WORK_DIR, self.LOGS_FOLDER, self.NAME)

    def _make_ckpt_name(self, epoch: int):
        return f"{self.NAME.replace("/", "_")}_{self.EPOCH_PREFIX}{epoch}.{self.CKPTS_EXT}"

    ### Config

    def __getitem__(self, field: str):
        return lookup(self.CONFIG, domain=self.NAME, field=field)

    def get_config(self, field: str, default: Any = None):
        try:
            return lookup(self.CONFIG, domain=self.NAME, field=field)
        except KeyError:
            logger.warning(f"field {field!r} is not found in the config, "
                           f"fallback to default {default!r}")
            return default

    def partial(self, func: Callable[..., _R], /, prefix: str):
        """Allow Sucrose to manage the parameters required for calling.

        Position-only args are not supported as the config data is stored in a dict.
        """
        return config_from_data(func, prefix, domain=self.NAME, data=self.CONFIG)

    ### Training

    def step(self, num: int = 1, /):
        self._step += num

    @property
    def num_steps(self): return self._step
    @num_steps.setter
    def num_steps(self, step: int):
        if not isinstance(step, int):
            raise TypeError(f"Step should be an int, but got {step.__class__.__name__}.")
        self._step = step

    def epoch_range(self, num: int, /):
        start = self.LAST_EPOCH
        return range(start, start + num)

    ### Ckpts

    def load_state_dict(
        self,
        epoch: int | None = None,
        *,
        load_step: bool = True,
        loader_kwds: dict[str, Any] = {},
        **state_dict: SupportsStateDict
    ) -> dict[str, Any]:
        """Load state dict to objects.

        Args:
            epoch (int | None, optional): The epoch number of the checkpoint
                file to read. Use the biggest number found in the folder if `None`.
                Defaults to `None`.
            load_step (bool, optional): Read step info (if exists) from file into
                the scenario if `True`. Defaults to `True`.
            loader_kwds (Dict[str, Any], optional): Keyword args for the loader function
                like `torch.load`.

        Returns:
            Dict: Objects remaining in the dictionary after loading the state dict.

        Examples:
            ```
            obj.load_state_dict(model=model, optim=optim)
            ```
            This line is equivelant to:
            ```
            data = torch.load("path/to/checkpoint.pt")
            model.load_state_dict(data['model'])
            optim.load_state_dict(data['optim'])
            ph.num_step = data['step'] # this key is actually `sucrose.const.STEP_KEY`
            ```
        """
        if epoch is None:
            epoch = self.LAST_EPOCH
        file_name = self._make_ckpt_name(epoch)

        try:
            extra_data = load_state_dict_impl(
                self.CKPTS_DIR, file_name, loader_kwds=loader_kwds, **state_dict
            )
        except FileNotFoundError:
            logger.warning(f"No checkpoint found for epoch {epoch}. "
                           "Loading skipped")
            return {}

        if load_step and self.STEP_KEY in extra_data:
            self.num_steps = extra_data[self.STEP_KEY]

        return extra_data

    def save_state_dict(
        self,
        interval: int = 1,
        *,
        save_step: bool = True,
        **state_dict: SupportsStateDict
    ):
        """Save state dict to `WORK_DIR/CKPTS_FOLDER/Scenario/FILE_NAME`.

        Args:
            interval (int): The epoch increase compared to the last saved
                checkpoint file. Calling this function will increase the save
                counter of the scenario by 1. The save operation will be
                triggered only when the counter reaches the interval, and then
                the counter will be cleared.
            save_step (bool, optional): Let the checkpoint file include step info,
                which grows as the `sucrose.step()` function is called. Defaults to `True`.
            **state_dict (Any): Objects to save. Save state dicts if they support.

        Examples:
            ```
            sucrose.save_state_dict(10, model=model, optim=optim)
            ```
            This line may generate a checkpoint file like:
            ```
            {
                "model": {...} # state dict of the model
                "optim": {...} # state dict of the optimizer
                "step": 1000 # for example
            }
            ```
        """
        self._local_epoch += 1
        if self._local_epoch < interval:
            return None

        self._local_epoch = 0
        self.LAST_EPOCH += interval
        file_name = self._make_ckpt_name(self.LAST_EPOCH)

        if save_step:
            if self.STEP_KEY in state_dict:
                raise ValueError(f"Key {self.STEP_KEY!r} is reserved for step info.")
            state_dict[self.STEP_KEY] = self.num_steps

        return save_state_dict_impl(
            self.CKPTS_DIR, file_name, **state_dict
        )

    ### Logs

    def start_pytorch_tensorboard(self, **kwargs):
        return start_pytorch_tensorboard_impl(self.LOGS_DIR, **kwargs)


def get_current_scenario() -> Scenario:
    result = get_current('Scenario')

    if result is None:
        raise RuntimeError("Start a scenario before getting the current.")

    return result


def auto_get_scenario(scen: Scenario | None = None, /) -> Scenario:
    if scen is None:
        return get_current_scenario()
    else:
        return scen
