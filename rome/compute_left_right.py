from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .compute_kv import compute_kv
from .compute_v_delta import RomeComputeVDeltaHparams, compute_v_delta
from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting


def compute_left_right(
    model: PreTrainedModel,
    module: torch.nn.Module,
    rewriting: TokenizedRomeRewriting,
    prefixes: Iterable[np.ndarray],
    preservings: Iterable[TokenizedRomePreserving],
    hparams: RomeComputeVDeltaHparams,
    c_inv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes = tuple(prefixes)

    k, v = compute_kv(
        model,
        module,
        prefixes,
        rewriting)

    v_delta = compute_v_delta(
        model,
        module,
        prefixes,
        rewriting,
        preservings,
        hparams,
        v)

    if c_inv is not None:
        c_inv = c_inv.to(v)
        left = (c_inv @ k) / (k @ c_inv @ k)
    else:
        left = k / (k @ k)

    right = v_delta

    return left, right
