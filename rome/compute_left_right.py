from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .compute_kv import compute_kv
from .compute_v_delta import compute_v_delta
from .hparams import ROMEHyperParams
from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting
from .utils import nethook


def compute_left_right(
    hparams: ROMEHyperParams,
    model: PreTrainedModel,
    rewriting: TokenizedRomeRewriting,
    prefixes: Iterable[np.ndarray],
    preservings: Iterable[TokenizedRomePreserving],
    c_inv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes = tuple(prefixes)

    module_name = hparams.rewrite_module_tmp.format(hparams.layer)
    module = nethook.get_module(model, module_name)

    k, v = compute_kv(
        model,
        module,
        prefixes,
        rewriting)

    v_delta = compute_v_delta(
        hparams,
        model,
        module,
        prefixes,
        rewriting,
        preservings,
        v)

    if c_inv is not None:
        c_inv = c_inv.to(v)
        left = (c_inv @ k) / (k @ c_inv @ k)
    else:
        left = k / (k @ k)

    right = v_delta

    return left, right
