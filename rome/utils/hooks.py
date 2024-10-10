from contextlib import contextmanager
from typing import Any, Callable

import torch


class StopForward(Exception):
    pass


@contextmanager
def forward_input_hook(module: torch.nn.Module, func: Callable[[torch.nn.Module, Any], Any]):
    hook = module.register_forward_pre_hook(func)
    try:
        yield
    except StopForward:
        pass
    finally:
        hook.remove()


@contextmanager
def forward_output_hook(module: torch.nn.Module, func: Callable[[torch.nn.Module, Any, Any], Any]):
    hook = module.register_forward_hook(func)
    try:
        yield
    except StopForward:
        pass
    finally:
        hook.remove()


@contextmanager
def no_grad_from(*parameters: torch.nn.Parameter | torch.Tensor):
    changed_params = []
    for parameter in parameters:
        if parameter.requires_grad:
            parameter.requires_grad = False
            changed_params.append(parameter)
    try:
        yield
    finally:
        for parameter in changed_params:
            parameter.requires_grad = True
