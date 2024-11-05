from typing import Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline


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
