from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .hooks import StopForward, forward_output_hook
from .rewriting import TokenizedRomeRewriting


def compute_kv(
    model: PreTrainedModel,
    module: torch.nn.Module,
    prefixes: Iterable[np.ndarray],
    rewriting: TokenizedRomeRewriting,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes = tuple(prefixes)
    prompts = [
        np.concatenate([prefix_tokenized, rewriting.prompt])
        for prefix_tokenized in prefixes]
    prompts_subject_token_index = [
        len(prefix_tokenized) + rewriting.subject_tail - 1
        for prefix_tokenized in prefixes]

    num = 0
    k_sum = 0
    v_sum = 0

    for prompt_tokenized, subject_token_index in zip(prompts, prompts_subject_token_index):
        prompt_tokenized = torch.asarray(prompt_tokenized, dtype=torch.int64, device=model.device)
        prompt_tokenized = prompt_tokenized.unsqueeze(0)

        def hook_func(_, inputs: tuple[torch.Tensor], output: torch.Tensor):
            nonlocal num, k_sum, v_sum
            k = inputs[0][0, subject_token_index].clone()
            v = output[0, subject_token_index].clone()
            k_sum += k
            v_sum += v
            num += 1
            raise StopForward()

        with torch.no_grad(), forward_output_hook(module, hook_func):
            model(prompt_tokenized)

    assert isinstance(k_sum, torch.Tensor)
    assert isinstance(v_sum, torch.Tensor)

    k = k_sum / num
    v = v_sum / num
    return k, v
