import warnings
from typing import Iterable

import numpy as np
import torch
from transformers import Conv1D, PreTrainedModel

from .compute_left_right import compute_left_right
from .hparams import RomeHparams
from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting
from .utils import nethook


def apply_rome_to_model(
    model: PreTrainedModel,
    hparams: RomeHparams,
    rewriting: TokenizedRomeRewriting,
    prefixes: Iterable[np.ndarray],
    preservings: Iterable[TokenizedRomePreserving],
    c_inv: torch.Tensor | None = None,
):
    module_name = hparams.rewrite_module_tmp.format(hparams.layer)
    module = nethook.get_module(model, module_name)

    (left, right) = compute_left_right(model, module, rewriting, prefixes, preservings, hparams.v_delta, c_inv)
    apply_left_right_to_module(module, left, right)


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
