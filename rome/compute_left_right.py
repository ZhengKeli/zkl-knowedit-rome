import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_kv import compute_kv
from .compute_left import compute_left
from .compute_right import compute_right
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

    left = compute_left(
        hparams,
        model,
        tokenizer,
        stats_dir,
        k)

    right = compute_right(
        model,
        rewriting_tokenized,
        preservings_tokenized,
        hparams,
        hparams.layer,
        left,
        prefixes_tokenized,
        k, v)

    return left, right
