import numpy as np
import torch
from transformers import PreTrainedModel

from .hparams import ROMEHyperParams
from .rewriting import TokenizedRomeRewriting
from .utils import nethook
from .utils.hooks import StopForward, pre_forward_hook


def compute_k(
    hparams: ROMEHyperParams,
    model: PreTrainedModel,
    layer: int,
    prefixes: list[np.ndarray],
    rewriting: TokenizedRomeRewriting,
) -> torch.Tensor:
    prompts = [
        np.concatenate([prefix_tokenized, rewriting.prompt])
        for prefix_tokenized in prefixes]
    prompts_subject_token_index = [
        len(prefix_tokenized) + rewriting.subject_tail - 1
        for prefix_tokenized in prefixes]

    module_name = hparams.rewrite_module_tmp.format(layer)
    module = nethook.get_module(model, module_name)

    k_sum = None
    k_num = 0

    for prompt_tokenized, subject_token_index in zip(prompts, prompts_subject_token_index):
        prompt_tokenized = torch.asarray(prompt_tokenized, dtype=torch.int64, device=model.device)
        prompt_tokenized = prompt_tokenized.unsqueeze(0)

        def hook_func(_, inputs: tuple[torch.Tensor]):
            nonlocal k_sum, k_num
            k = inputs[0][0, subject_token_index].clone()
            if k_sum is None:
                k_sum = k
                k_num = 1
            else:
                k_sum += k
                k_num += 1
            raise StopForward()

        with torch.no_grad(), pre_forward_hook(module, hook_func):
            model(prompt_tokenized)

    assert isinstance(k_sum, torch.Tensor)
    return k_sum / k_num
