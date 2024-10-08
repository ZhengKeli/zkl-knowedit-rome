from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_left_right import compute_left_right
from .hparams import ROMEHyperParams
from .prefixes import iter_random_prefixes
from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting
from .utils import nethook


def apply_rome_to_model(
    model: PreTrainedModel,
    hparams: ROMEHyperParams,
    rewriting: TokenizedRomeRewriting,
    prefixes: Iterable[np.ndarray],
    preservings: Iterable[TokenizedRomePreserving],
    c_inv: torch.Tensor | None = None,
):
    weight_name = f"{hparams.rewrite_module_tmp.format(hparams.layer)}.weight"
    weight = nethook.get_parameter(model, weight_name)

    (left, right) = compute_left_right(hparams, model, rewriting, prefixes, preservings, c_inv)

    with torch.no_grad():
        delta_weight = torch.outer(left, right)
        weight[...] += delta_weight


def make_default_prefixes(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> tuple[np.ndarray, ...]:
    prefixes = tuple(iter_random_prefixes(model, tokenizer, [(5, 10), (10, 10)]))
    prefixes_tokenized = tuple(np.asarray(tokenizer.encode(prefix), dtype=np.int64) for prefix in prefixes)
    return prefixes_tokenized


def make_default_preservings(
    tokenizer: PreTrainedTokenizer,
    rewriting: TokenizedRomeRewriting,
) -> tuple[TokenizedRomePreserving, ...]:
    preserving = TokenizedRomePreserving(
        prompt=np.concatenate([
            rewriting.subject,
            tokenizer.encode(" is a")]),
        subject_head=0,
        subject_tail=len(rewriting.subject))
    return preserving,
