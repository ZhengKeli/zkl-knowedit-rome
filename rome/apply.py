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

    weight_name = f"{hparams.rewrite_module_tmp.format(hparams.layer)}.weight"
    weight = nethook.get_parameter(model, weight_name)

    (left, right) = compute_left_right(model, module, rewriting, prefixes, preservings, hparams.v_delta, c_inv)

    with torch.no_grad():
        delta_weight = torch.outer(left, right)
        weight[...] += delta_weight
