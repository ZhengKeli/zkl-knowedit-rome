import os
from typing import Iterable, Literal

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .apply_left_right import apply_left_right
from .compute_c import ComputeCCallback, ComputeCHparams, ComputeCMetrics
from .compute_c_inv import compute_c_inv
from .compute_left_right import compute_left_right
from .compute_v_delta import ComputeVDeltaCallback, ComputeVDeltaHparams, ComputeVDeltaMetrics
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
    compute_c_samples: Iterable[str | np.ndarray] | Literal['wikipedia'] | None = 'wikipedia',
    compute_c_hparams: ComputeCHparams | None = None,
    compute_c_callback: ComputeCCallback | Literal['tqdm'] | None = 'tqdm',
    cache_c_inv_file_path: os.PathLike | str | None = None,
    compute_v_delta_hparams: ComputeVDeltaHparams,
    compute_v_delta_callback: ComputeVDeltaCallback | Literal['tqdm'] | None = 'tqdm',
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

        compute_c_samples = map(tokenize, compute_c_samples)

    if compute_c_samples == 'wikipedia':
        compute_c_samples = WikipediaComputeCSamples(tokenizer=tokenizer)

    if compute_c_callback == 'tqdm':
        compute_c_callback = TqdmComputeCCallback()
    assert isinstance(compute_c_callback, ComputeCCallback | None)

    if compute_v_delta_callback == 'tqdm':
        compute_v_delta_callback = TqdmComputeVDeltaCallback()
    assert isinstance(compute_v_delta_callback, ComputeVDeltaCallback | None)

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
    compute_c_samples: Iterable[np.ndarray] | None = None,
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


class WikipediaComputeCSamples(Iterable[str | np.ndarray]):
    def __init__(self, *,
        path: os.PathLike | str = "wikipedia",
        name: str = "20220301.en",
        split: Literal['train', 'validation', 'test'] = "train",
        streaming: bool = True,
        tokenizer: PreTrainedTokenizer | None = None
    ):
        self.path = path
        self.name = name
        self.split = split
        self.streaming = streaming
        self.tokenizer = tokenizer

    def __iter__(self):
        dataset = load_dataset(
            self.path,
            self.name,
            split=self.split,
            streaming=self.streaming)

        for sample in dataset:
            sample = sample["text"]
            if self.tokenizer is not None:
                sample = self.tokenizer.encode(sample)
                sample = np.asarray(sample, dtype=np.int64)
            yield sample


class TqdmComputeCCallback(ComputeCCallback):
    def __init__(self):
        self.progressbar: tqdm | None = None

    def on_start(self, hparams: ComputeCHparams):
        self.progressbar = tqdm(
            desc="Computing c",
            total=hparams.stopping_tokens_num)

    def on_batch(self, metrics: ComputeCMetrics):
        self.progressbar.update(metrics.processed_tokens_num - self.progressbar.n)

    def on_stop(self, metrics: ComputeCMetrics):
        self.progressbar.close()


class TqdmComputeVDeltaCallback(ComputeVDeltaCallback):
    def __init__(self):
        self.progressbar: tqdm | None = None

    def on_start(self, hparams: ComputeVDeltaHparams):
        self.progressbar = tqdm(
            desc="Computing v_delta",
            total=hparams.stopping_steps_num)

    def on_step(self, metrics: ComputeVDeltaMetrics):
        self.progressbar.update(metrics.processed_steps_num - self.progressbar.n)
        self.progressbar.set_postfix_str(", ".join([
            f"loss={metrics.loss.item():.4f}",
            f"rewriting_acc={metrics.rewriting_acc.mean().item():.4f}",
            f"rewriting_loss={metrics.rewriting_loss.item():.4f}",
            f"preserving_loss={metrics.preserving_loss.item():.4f}",
            f"regularization_loss={metrics.regularization_loss.item():.4f}"]))

    def on_stop(self, metrics: ComputeVDeltaMetrics):
        self.progressbar.close()
