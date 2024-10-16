import os.path
import sys
from typing import Callable, Iterable

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline

from zkl_rome import ComputeCHparams, ComputeVDeltaMetrics, compute_c


def load_dataset_for_compute_c():
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train")
    next(iter(dataset))
    return dataset


def iter_samples_for_compute_c(dataset: Dataset, tokenizer: PreTrainedTokenizer):
    for sample in dataset:
        sample = sample["text"]
        sample = tokenizer.encode(sample)
        sample = np.asarray(sample, dtype=np.int64)
        yield sample


def compute_c_inv(
    hparams: ComputeCHparams,
    model: PreTrainedModel,
    module: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
) -> torch.Tensor:
    dataset = load_dataset_for_compute_c()
    iterator = iter_samples_for_compute_c(dataset, tokenizer)
    c = compute_c(hparams, model, module, iterator)
    c_inv = torch.inverse(c)
    return c_inv


def caching_torch_tensor(cache_file_path: str, device: torch.device | str | None = None):
    def decorator(func: Callable[[...], torch.Tensor]):
        def wrapper(*args, **kwargs):
            try:
                print(f"Loading cache from {cache_file_path}.", file=sys.stderr)
                tensor = torch.load(cache_file_path)
                print(f"Loaded.", file=sys.stderr)
            except Exception:
                print(f"Failed.", file=sys.stderr)

                print(f"Computing instead...", file=sys.stderr)
                tensor = func(*args, **kwargs)
                print(f"Computed.", file=sys.stderr)

                print(f"Saving cache to {cache_file_path}", file=sys.stderr)
                os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                torch.save(tensor, cache_file_path)
                print(f"Saved.", file=sys.stderr)
            return tensor.to(device)

        return wrapper

    return decorator


def generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: Iterable[str]):
    pipe = pipeline("text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        num_return_sequences=1,
        return_full_text=True,
        max_new_tokens=64)
    return tuple(pipe(prompt)[0]['generated_text'] for prompt in prompts)


def print_v_delta_metrics(metrics: ComputeVDeltaMetrics):
    print(", ".join([
        f"step={metrics.step}",
        f"loss={metrics.loss.item():.4f}",
        f"rewriting_acc={metrics.rewriting_acc.mean().item():.4f}",
        f"rewriting_loss={metrics.rewriting_loss.item():.4f}",
        f"preserving_loss={metrics.preserving_loss.item():.4f}",
        f"regularization_loss={metrics.regularization_loss.item():.4f}"]))
