import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_c import compute_c_inv
from .compute_k import compute_k
from .hparams import ROMEHyperParams
from .rewriting import TextRomeRewriting, TokenizedRomeRewriting


def compute_u(
    hparams: ROMEHyperParams,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prefixes: list[str],
    rewriting: TextRomeRewriting,
    stats_dir: str,
) -> torch.Tensor:
    layer = hparams.layer

    rewriting_tokenized = TokenizedRomeRewriting.from_text_rewriting(rewriting, tokenizer)
    prefixes_tokenized = [np.asarray(tokenizer.encode(prefix), dtype=np.int64) for prefix in prefixes]
    u = compute_k(
        hparams,
        model,
        layer,
        prefixes_tokenized,
        rewriting_tokenized)

    # Apply inverse second moment adjustment
    if hparams.mom2_adjustment:
        c_inv = compute_c_inv(
            model,
            tokenizer,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            stats_dir)
        c_inv = c_inv.to(u)
        u = c_inv @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
