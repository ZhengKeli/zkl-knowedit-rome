from typing import Iterable

import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline

from zkl_rome import ComputeVDeltaMetrics


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
