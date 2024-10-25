from typing import Callable, Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .compute_k_v import compute_k_v
from .compute_v_delta import ComputeVDeltaHparams, ComputeVDeltaMetrics, compute_v_delta
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
    compute_v_delta_callback: Callable[[ComputeVDeltaMetrics], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes = tuple(prefixes)

    k, v = compute_k_v(
        model,
        module,
        prefixes,
        rewriting)

    v_delta = compute_v_delta(
        compute_v_delta_hparams,
        model,
        module,
        prefixes,
        rewriting,
        preservings,
        v,
        callback=compute_v_delta_callback)

    if c_inv is not None:
        c_inv = c_inv.to(v)
        left = (c_inv @ k) / (k @ c_inv @ k)
    else:
        left = k / (k @ k)

    right = v_delta

    return left, right
