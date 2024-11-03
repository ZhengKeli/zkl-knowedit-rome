import os
from typing import Callable, Iterable

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .apply_left_right import apply_left_right
from .compute_c import ComputeCCallback, ComputeCHparams
from .compute_c_inv import compute_c_inv
from .compute_left_right import compute_left_right
from .compute_v_delta import ComputeVDeltaCallback, ComputeVDeltaHparams
from .generate_prefixes import GeneratePrefixesHparams, generate_prefixes
from .preserving import TextPreserving, TokenizedPreserving
from .rewriting import TextRewriting, TokenizedRewriting


def rome(*,
    apply: bool = True,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | None,
    module_name: str | None = None,
    rewriting: TextRewriting | TokenizedRewriting,
    prefixes: Iterable[str | np.ndarray] | None = None,
    preservings: Iterable[TextPreserving | TokenizedPreserving] | None = None,
    compute_c_samples: Iterable[str | np.ndarray] | Callable[[], Iterable[str | np.ndarray]] | None = None,
    compute_c_hparams: ComputeCHparams | None = None,
    compute_c_callback: ComputeCCallback | None = None,
    cache_c_inv_file_path: os.PathLike | str | None = None,
    compute_v_delta_hparams: ComputeVDeltaHparams,
    compute_v_delta_callback: ComputeVDeltaCallback | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(rewriting, TextRewriting):
        rewriting = rewriting.tokenize(tokenizer)

    if prefixes is None:
        prefixes = generate_prefixes_by_default(model, tokenizer)
    else:
        def tokenize(prefix: str | np.ndarray):
            if isinstance(prefix, str):
                prefix = tokenizer.encode(prefix)
            prefix = np.asarray(prefix, dtype=np.int64)
            return prefix

        prefixes = map(tokenize, prefixes)

    if preservings is None:
        preservings = generate_preservings_by_default(tokenizer, rewriting)
    else:
        def tokenize(preserving: TextPreserving | TokenizedPreserving):
            if isinstance(preserving, TextPreserving):
                preserving = preserving.tokenize(tokenizer)
            return preserving

        preservings = map(tokenize, preservings)

    if compute_c_samples is not None:
        def tokenize(sample: str | np.ndarray):
            if isinstance(sample, str):
                sample = tokenizer.encode(sample)
            sample = np.asarray(sample, dtype=np.int64)
            return sample

        if isinstance(compute_c_samples, Callable):
            compute_c_samples_callable = compute_c_samples
            compute_c_samples = lambda: map(tokenize, compute_c_samples_callable())
        else:
            compute_c_samples = map(tokenize, compute_c_samples)

    module = model.get_submodule(module_name)

    c_inv = load_or_compute_c_inv(
        model=model,
        module=module,
        compute_c_samples=compute_c_samples,
        compute_c_hparams=compute_c_hparams,
        compute_c_callback=compute_c_callback,
        cache_c_inv_file_path=cache_c_inv_file_path)

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


def load_or_compute_c_inv(*,
    model: PreTrainedModel,
    module: torch.nn.Module,
    compute_c_samples: Iterable[np.ndarray] | Callable[[], Iterable[np.ndarray]] | None = None,
    compute_c_hparams: ComputeCHparams | None = None,
    compute_c_callback: ComputeCCallback | None = None,
    cache_c_inv_file_path: os.PathLike | str | None = None,
) -> torch.Tensor | None:
    c_inv = None
    if cache_c_inv_file_path is not None:
        try:
            c_inv = torch.load(cache_c_inv_file_path, map_location=model.device)
        except IOError:
            pass
    if c_inv is None and compute_c_samples is not None and compute_c_hparams is not None:
        if isinstance(compute_c_samples, Callable):
            compute_c_samples = compute_c_samples()
        c_inv = compute_c_inv(
            model=model,
            module=module,
            compute_c_samples=compute_c_samples,
            compute_c_hparams=compute_c_hparams,
            compute_c_callback=compute_c_callback)
        if cache_c_inv_file_path is not None:
            os.makedirs(os.path.dirname(cache_c_inv_file_path), exist_ok=True)
            torch.save(c_inv, cache_c_inv_file_path)
    return c_inv
