import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_c import compute_c_inv
from .compute_k import compute_k
from .hparams import ROMEHyperParams
from .rewriting import TokenizedRomeRewriting


def compute_left(
    hparams: ROMEHyperParams,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prefixes: list[np.ndarray],
    rewriting: TokenizedRomeRewriting,
    stats_dir: str,
) -> torch.Tensor:
    k = compute_k(
        hparams,
        model,
        hparams.layer,
        prefixes,
        rewriting)

    # Apply inverse second moment adjustment
    if hparams.mom2_adjustment:
        c_inv = compute_c_inv(
            model,
            tokenizer,
            hparams.rewrite_module_tmp.format(hparams.layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            stats_dir).to(k)
        left = c_inv @ k
    else:
        left = k

    return left / left.norm()
