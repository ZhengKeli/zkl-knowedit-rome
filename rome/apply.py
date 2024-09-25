import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .prefixes import iter_random_prefixes
from .preserving import TextRomePreserving, TokenizedRomePreserving
from .rewriting import TextRomeRewriting, TokenizedRomeRewriting
from .utils import nethook
from .compute_left_right import compute_left_right
from .hparams import ROMEHyperParams


def apply_rome_to_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    rewritings: list[TextRomeRewriting],
    hparams: ROMEHyperParams,
    c_inv: torch.Tensor | None = None,
):
    prefixes = list(iter_random_prefixes(model, tokenizer, hparams.context_template_length_params))
    prefixes_tokenized = [np.asarray(tokenizer.encode(prefix), dtype=np.int64) for prefix in prefixes]

    for i, rewriting in enumerate(rewritings):
        preservings = [TextRomePreserving(
            prompt=f"{rewriting.subject} is a ",
            subject_head=0,
            subject_tail=len(rewriting.subject)
        )]

        rewriting_tokenized = TokenizedRomeRewriting.from_text_rewriting(rewriting, tokenizer)
        preservings_tokenized = [
            TokenizedRomePreserving.from_text_preserving(preserving, tokenizer)
            for preserving in preservings]

        weight_name = f"{hparams.rewrite_module_tmp.format(hparams.layer)}.weight"
        weight = nethook.get_parameter(model, weight_name)

        (left, right) = compute_left_right(model, rewriting_tokenized, prefixes_tokenized, preservings_tokenized, hparams, c_inv)

        with torch.no_grad():
            delta_weight = torch.outer(left, right)
            weight[...] += delta_weight
