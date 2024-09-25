import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .rewriting import TextRomeRewriting
from .utils import nethook
from .compute_left_right import compute_left_right
from .hparams import ROMEHyperParams


def apply_rome_to_model(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    rewritings: list[TextRomeRewriting],
    hparams: ROMEHyperParams,
    stats_dir: str,
):
    for i, rewriting in enumerate(rewritings):
        weight_name = f"{hparams.rewrite_module_tmp.format(hparams.layer)}.weight"
        weight = nethook.get_parameter(model, weight_name)

        (left, right) = compute_left_right(model, tok, rewriting, hparams, stats_dir)

        with torch.no_grad():
            delta_weight = torch.outer(left, right)
            weight[...] += delta_weight
