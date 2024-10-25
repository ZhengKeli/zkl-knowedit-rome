from typing import Callable, Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .compute_c import ComputeCHparams, ComputeCMetrics, compute_c


def compute_c_inv(*,
    model: PreTrainedModel,
    module: torch.nn.Module,
    compute_c_samples: Iterable[np.ndarray],
    compute_c_hparams: ComputeCHparams,
    compute_c_callback: Callable[[ComputeCMetrics], None] | None = None,
) -> torch.Tensor:
    c = compute_c(
        model=model,
        module=module,
        samples=compute_c_samples,
        hparams=compute_c_hparams,
        callback=compute_c_callback)
    c_inv = torch.inverse(c)
    return c_inv
