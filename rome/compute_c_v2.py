from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from rome.utils.batching import iter_by_batch
from rome.utils.hooks import StopForward, forward_input_hook


@dataclass(kw_only=True)
class RomeComputeCHParams:
    total_tokens_num: int | None = None
    batch_samples_num: int
    context_tokens_num: int


def compute_c(
    hparams: RomeComputeCHParams,
    model: PreTrainedModel,
    module: torch.nn.Module,
    dataset: Iterable[np.ndarray],
    verbose: bool = True
):
    iterator = iter_by_batch(dataset,
        batch_size=hparams.batch_samples_num,
        batch_len=hparams.context_tokens_num,
        return_mask=True)

    if verbose:
        progress_bar = tqdm(total=hparams.total_tokens_num)

    c_sum = 0
    c_num = 0

    for batch_tokens, batch_masks in iterator:
        if hparams.total_tokens_num is not None:
            if c_num >= hparams.total_tokens_num:
                break

        def hook(_, inputs):
            nonlocal c_num, c_sum
            ks = inputs[0]
            ks = ks.reshape([-1, ks.shape[-1]])
            ms = batch_masks.reshape([-1])
            ks = ks[ms]
            ks = ks.to(torch.float32)

            n = torch.sum(ms, dtype=torch.int64).cpu().item()
            c = torch.matmul(ks.T, ks)

            if verbose:
                progress_bar.update(n)

            c_num += n
            c_sum += c.clone()
            raise StopForward

        batch_tokens = torch.from_numpy(batch_tokens).to(device=model.device, dtype=torch.int64)
        batch_masks = torch.from_numpy(batch_masks).to(device=model.device, dtype=torch.bool)
        with torch.no_grad(), forward_input_hook(module, hook):
            model(batch_tokens, attention_mask=batch_masks)

    return c_sum / c_num