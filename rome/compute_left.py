import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_c import compute_c_inv
from .hparams import ROMEHyperParams


def compute_left(
    hparams: ROMEHyperParams,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    stats_dir: str,
    k: torch.Tensor,
) -> torch.Tensor:
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
