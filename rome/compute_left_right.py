from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .compute_kv import compute_kv
from .compute_v_delta import compute_v_delta
from .hparams import ROMEHyperParams
from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting


def compute_left_right(
    model: PreTrainedModel,
    rewriting_tokenized: TokenizedRomeRewriting,
    prefixes_tokenized: Iterable[np.ndarray],
    preservings_tokenized: Iterable[TokenizedRomePreserving],
    hparams: ROMEHyperParams,
    c_inv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k, v = compute_kv(
        hparams,
        model,
        hparams.layer,
        prefixes_tokenized,
        rewriting_tokenized)

    v_delta = compute_v_delta(
        hparams,
        model,
        hparams.layer,
        prefixes_tokenized,
        rewriting_tokenized,
        preservings_tokenized,
        v)

    if c_inv is not None:
        c_inv = c_inv.to(v)
        left = (c_inv @ k) / (k @ c_inv @ k)
    else:
        left = k / (k @ k)

    right = v_delta

    return left, right
