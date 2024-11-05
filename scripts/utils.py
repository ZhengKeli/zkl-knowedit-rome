from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline


def iter_compute_c_samples_from_wikipedia(tokenizer: PreTrainedTokenizer | None = None):
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True)

    for sample in dataset:
        sample = sample["text"]
        if tokenizer is not None:
            sample = tokenizer.encode(sample)
            sample = np.asarray(sample, dtype=np.int64)
        yield sample


def generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: Iterable[str]):
    pipe = pipeline("text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        truncation=True,
        max_new_tokens=64)
    return tuple(pipe(prompt)[0]['generated_text'] for prompt in prompts)


def compute_cosine_similarity(a1: torch.Tensor, a2: torch.Tensor):
    a1 = torch.reshape(a1, [-1])
    a2 = torch.reshape(a2, [-1])
    return torch.nn.functional.cosine_similarity(a1, a2, dim=0)
