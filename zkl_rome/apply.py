import warnings

import torch
from transformers import Conv1D


def apply_left_right_to_module(
    module: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
):
    if isinstance(module, Conv1D):
        module.weight.data.add_(torch.outer(left, right))
    elif isinstance(module, (torch.nn.Linear, torch.nn.LazyLinear)):
        module.weight.data.add_(torch.outer(right, left))
    else:
        def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            k = inputs[0]
            v_delta = torch.sum(k * left, dim=-1, keepdim=True) * right
            return output + v_delta

        module.register_forward_hook(hook)
        warnings.warn(f"Not recognized module type {module}, fallback to use hook.")
