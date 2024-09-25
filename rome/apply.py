from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .compute_left_right import compute_left_right
from .hparams import ROMEHyperParams
from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting
from .utils import nethook


def apply_rome_to_model(
    model: PreTrainedModel,
    hparams: ROMEHyperParams,
    rewriting: TokenizedRomeRewriting,
    prefixes: Iterable[np.ndarray],
    preservings: Iterable[TokenizedRomePreserving],
    c_inv: torch.Tensor | None = None,
):
    module_name = hparams.rewrite_module_tmp.format(hparams.layer)
    module = nethook.get_module(model, module_name)

    (left, right) = compute_left_right(model, module, rewriting, prefixes, preservings, hparams.v_delta, c_inv)
    weight_delta = torch.outer(left, right)

    apply_weight_delta(module, weight_delta)


def apply_weight_delta(
    module: torch.nn.Module,
    weight_delta: torch.Tensor,
):
    with torch.no_grad():
        module.weight[...] += weight_delta
