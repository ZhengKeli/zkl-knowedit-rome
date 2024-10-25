import os
from typing import Callable, Iterable

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from zkl_rome import GeneratePrefixesHparams, TokenizedPreserving, TokenizedRewriting, generate_prefixes
from .apply_left_right import apply_left_right
from .compute_c import ComputeCHparams
from .compute_c_inv import compute_c_inv
from .compute_left_right import compute_left_right
from .compute_v_delta import ComputeVDeltaHparams, ComputeVDeltaMetrics


def rome(*,
    apply: bool = True,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None,
    module_name: str | None = None,
    rewriting: TokenizedRewriting,
    prefixes: Iterable[np.ndarray] | None = None,
    preservings: Iterable[TokenizedPreserving] | None = None,
    compute_c_dataset: Iterable[np.ndarray] | None = None,
    compute_c_hparams: ComputeCHparams | None = None,
    compute_c_callback: Callable[[ComputeCHparams], None] | None = None,
    cache_c_inv_file_path: os.PathLike | str | None = None,
    compute_v_delta_hparams: ComputeVDeltaHparams,
    compute_v_delta_callback: Callable[[ComputeVDeltaMetrics], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = model.get_submodule(module_name)

    c_inv = load_or_compute_c_inf(
        model=model,
        module=module,
        compute_c_dataset=compute_c_dataset,
        compute_c_hparams=compute_c_hparams,
        compute_c_callback=compute_c_callback,
        cache_c_inv_file_path=cache_c_inv_file_path)

    if prefixes is None:
        prefixes = generate_prefixes_by_default(model, tokenizer)

    if preservings is None:
        preservings = generate_preservings_by_default(tokenizer, rewriting)

    (left, right) = compute_left_right(
        model=model,
        module=module,
        prefixes=prefixes,
        rewriting=rewriting,
        preservings=preservings,
        c_inv=c_inv,
        compute_v_delta_hparams=compute_v_delta_hparams,
        compute_v_delta_callback=compute_v_delta_callback)

    if apply:
        apply_left_right(module, left, right)

    return left, right


def load_or_compute_c_inf(*,
    model: PreTrainedModel,
    module: torch.nn.Module,
    compute_c_dataset: Iterable[np.ndarray] | None = None,
    compute_c_hparams: ComputeCHparams | None = None,
    compute_c_callback: Callable[[ComputeCHparams], None] | None = None,
    cache_c_inv_file_path: os.PathLike | str | None = None,
) -> torch.Tensor | None:
    c_inv = None
    if cache_c_inv_file_path is not None:
        try:
            c_inv = torch.load(cache_c_inv_file_path, map_location=model.device)
        except IOError:
            pass
    if c_inv is None and compute_c_dataset is not None and compute_c_hparams is not None:
        c_inv = compute_c_inv(
            model=model,
            module=module,
            compute_c_dataset=compute_c_dataset,
            compute_c_hparams=compute_c_hparams,
            compute_c_callback=compute_c_callback)
    return c_inv


def generate_prefixes_by_default(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> tuple[np.ndarray, ...]:
    return generate_prefixes(
        model=model,
        tokenizer=tokenizer,
        hparams=(
            GeneratePrefixesHparams(
                seperator=". ",
                num_tokens=5,
                num_sequences=10),
            GeneratePrefixesHparams(
                seperator=". ",
                num_tokens=10,
                num_sequences=10)))


def generate_preservings_by_default(
    tokenizer: PreTrainedTokenizer,
    rewriting: TokenizedRewriting,
) -> tuple[TokenizedPreserving, ...]:
    preserving = TokenizedPreserving(
        prompt=np.concatenate([
            rewriting.subject,
            tokenizer.encode(" is a")]),
        subject_head=0,
        subject_tail=len(rewriting.subject))
    return preserving,
