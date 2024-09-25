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
    left_vector: torch.Tensor,
    prefixes: list[np.ndarray],
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    v_delta = compute_v_delta(
        hparams,
        model,
        layer,
        prefixes,
        rewriting,
        preservings,
        v)

    v_star = v + v_delta

    # Solving the linear system to compute the right vector
    right_vector = v_delta / torch.dot(k, left_vector)
    print(f"Delta norm: {(v_star - v).norm().item()}")
    print(f"Change in target norm: {v.norm().item()} to {v_star.norm().item()} => {(v_star.norm() - v.norm()).item()}")
    print(f"Division Factor: {torch.dot(k, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector
