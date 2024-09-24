import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_u import compute_u
from .compute_v_v2 import compute_v
from .hparams import ROMEHyperParams
from .prefixes import iter_random_prefixes
from .preserving import TextRomePreserving, TokenizedRomePreserving
from .rewriting import TextRomeRewriting, TokenizedRomeRewriting


def execute_rome(
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

    u = compute_u(
        hparams,
        model,
        tokenizer,
        prefixes_tokenized,
        rewriting_tokenized,
        stats_dir)

    v = compute_v(
        model,
        rewriting_tokenized,
        preservings_tokenized,
        hparams,
        hparams.layer,
        u,
        prefixes_tokenized)

    return u, v
