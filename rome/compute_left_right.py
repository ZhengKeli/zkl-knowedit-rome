import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_c import compute_c_inv
from .compute_kv import compute_kv
from .compute_v_delta import compute_v_delta
from .hparams import ROMEHyperParams
from .prefixes import iter_random_prefixes
from .preserving import TextRomePreserving, TokenizedRomePreserving
from .rewriting import TextRomeRewriting, TokenizedRomeRewriting


def compute_left_right(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    rewriting: TextRomeRewriting,
    hparams: ROMEHyperParams,
    stats_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes = list(iter_random_prefixes(model, tokenizer, hparams.context_template_length_params))
    prefixes_tokenized = [np.asarray(tokenizer.encode(prefix), dtype=np.int64) for prefix in prefixes]
    rewriting_tokenized = TokenizedRomeRewriting.from_text_rewriting(rewriting, tokenizer)

    preservings = [TextRomePreserving(
        prompt=f"{rewriting.subject} is a ",
        subject_head=0,
        subject_tail=len(rewriting.subject)
    )]
    preservings_tokenized = [
        TokenizedRomePreserving.from_text_preserving(preserving, tokenizer)
        for preserving in preservings]

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

    if hparams.mom2_adjustment:
        c_inv = compute_c_inv(
            model,
            tokenizer,
            hparams.rewrite_module_tmp.format(hparams.layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            stats_dir).to(k)
        left = (c_inv @ k) / (k @ c_inv @ k)
    else:
        left = k / (k @ k)

    right = v_delta

    return left, right
