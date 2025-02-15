
import threading

__all__ = [
    "CKPTS_FOLDER",
    "LOGS_FOLDER",

    "MODEL_KEY",
    "OPTIM_KEY",
    "EXTRA_KEY",
    "STEP_KEY",

    "set_current",
    "get_current"
]


CKPTS_FOLDER = 'ckpts'
LOGS_FOLDER = 'logs'

MODEL_KEY = 'model'
OPTIM_KEY = 'optim'
EXTRA_KEY = 'extra'
STEP_KEY = 'step'

LOCAL_THREAD = threading.local()


def set_current(key: str, data):
    setattr(LOCAL_THREAD, key, data)

def get_current(key: str):
    if hasattr(LOCAL_THREAD, key):
        return getattr(LOCAL_THREAD, key)
    else:
        return None
