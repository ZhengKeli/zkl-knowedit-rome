from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .batching import iter_by_batch
from .hooks import StopForward, forward_input_hook


@dataclass(kw_only=True)
class ComputeCHparams:
    total_tokens_num: int | None = None
    batch_samples_num: int
    context_tokens_num: int


@dataclass(kw_only=True)
class ComputeCMetrics:
    tokens: int


def compute_c(*,
    model: PreTrainedModel,
    module: torch.nn.Module,
    dataset: Iterable[np.ndarray],
    hparams: ComputeCHparams,
    callback: Callable[[ComputeCMetrics], None] | None = None,
) -> torch.Tensor:
    iterator = iter_by_batch(dataset,
        batch_size=hparams.batch_samples_num,
        batch_len=hparams.context_tokens_num,
        return_mask=True)

    c_sum = 0
    c_num = 0

    for batch_tokens, batch_masks in iterator:
        if hparams.total_tokens_num is not None:
            if c_num >= hparams.total_tokens_num:
                break

        def hook(_, inputs):
            ks = inputs[0]
            ks = ks.reshape([-1, ks.shape[-1]])
            ms = batch_masks.reshape([-1])
            ks = ks[ms]
            ks = ks.to(torch.float32)

            c = torch.matmul(ks.T, ks)
            n = torch.sum(ms, dtype=torch.int64)

            c = c.clone()
            n = n.cpu().item()

            nonlocal c_sum, c_num
            c_sum += c
            c_num += n

            if callback is not None:
                callback(ComputeCMetrics(tokens=c_num))

            raise StopForward

        batch_tokens = torch.from_numpy(batch_tokens).to(device=model.device, dtype=torch.int64)
        batch_masks = torch.from_numpy(batch_masks).to(device=model.device, dtype=torch.bool)
        with torch.no_grad(), forward_input_hook(module, hook):
            model(batch_tokens, attention_mask=batch_masks)

    if not isinstance(c_sum, torch.Tensor):
        raise ValueError("At least one sample must be processed to compute C!")

    return c_sum / c_num
