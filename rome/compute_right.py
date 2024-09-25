from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .compute_v_delta import compute_v_delta
from .hparams import ROMEHyperParams
from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting


def compute_right(
    model: PreTrainedModel,
    rewriting: TokenizedRomeRewriting,
    preservings: Iterable[TokenizedRomePreserving],
    hparams: ROMEHyperParams,
    layer: int,
    prefixes: list[np.ndarray],
    v: torch.Tensor,
) -> torch.Tensor:
    return compute_v_delta(
        hparams,
        model,
        layer,
        prefixes,
        rewriting,
        preservings,
        v)
