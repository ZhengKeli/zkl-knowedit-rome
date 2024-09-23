from contextlib import contextmanager
from typing import Any, Callable

import torch


class StopForward(Exception):
    pass


@contextmanager
def pre_forward_hook(module: torch.nn.Module, func: Callable[[torch.nn.Module, Any], Any]):
    hook = module.register_forward_pre_hook(func)
    try:
        yield
    except StopForward:
        pass
    finally:
        hook.remove()


@contextmanager
def post_forward_hook(module: torch.nn.Module, func: Callable[[torch.nn.Module, Any, Any], Any]):
    hook = module.register_forward_hook(func)
    try:
        yield
    except StopForward:
        pass
    finally:
        hook.remove()
