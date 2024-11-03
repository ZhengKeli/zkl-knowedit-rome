from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .compute_k_v import compute_k_v
from .compute_v_delta import ComputeVDeltaCallback, ComputeVDeltaHparams, compute_v_delta
from .preserving import TokenizedPreserving
from .rewriting import TokenizedRewriting


def compute_left_right(*,
    model: PreTrainedModel,
    module: torch.nn.Module,
    prefixes: Iterable[np.ndarray],
    rewriting: TokenizedRewriting,
    preservings: Iterable[TokenizedPreserving],
    c_inv: torch.Tensor | None = None,
    compute_v_delta_hparams: ComputeVDeltaHparams,
    compute_v_delta_callback: ComputeVDeltaCallback | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes = tuple(prefixes)

    k, v = compute_k_v(
        model=model,
        module=module,
        prefixes=prefixes,
        rewriting=rewriting)

    v_delta = compute_v_delta(
        model=model,
        module=module,
        prefixes=prefixes,
        rewriting=rewriting,
        preservings=preservings,
        v=v,
        hparams=compute_v_delta_hparams,
        callback=compute_v_delta_callback)

    if c_inv is not None:
        c_inv = c_inv.to(v)
        left = (c_inv @ k) / (k @ c_inv @ k)
    else:
        left = k / (k @ k)

    right = v_delta

    return left, right
