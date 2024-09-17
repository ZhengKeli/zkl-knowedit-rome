import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from util import nethook
from .compute_uv import execute_rome
from .hparams import ROMEHyperParams


def apply_rome_to_model(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    requests: list[dict],
    hparams: ROMEHyperParams,
    stats_dir: str,
):
    for i, request in enumerate(requests):
        weight_name = f"{hparams.rewrite_module_tmp.format(hparams.layer)}.weight"
        weight = nethook.get_parameter(model, weight_name)

        (delta_u, delta_v) = execute_rome(model, tok, request, hparams, stats_dir)

        with torch.no_grad():
            delta_weight = torch.outer(delta_u, delta_v)
            weight[...] += delta_weight
