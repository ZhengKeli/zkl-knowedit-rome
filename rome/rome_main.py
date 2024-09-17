from copy import deepcopy

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from util import nethook
from .compute_u import compute_u
from .compute_v import compute_v
from .hparams import ROMEHyperParams
from .prefixes import get_context_templates


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


def execute_rome(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    request: dict,
    hparams: ROMEHyperParams,
    stats_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # prefixes
    context_templates = get_context_templates(model, tok, hparams.context_template_length_params)

    # Compute rank-1 update matrix
    left_vector: torch.Tensor = compute_u(
        model,
        tok,
        request,
        hparams,
        hparams.layer,
        context_templates,
        stats_dir,
    )
    print("Left vector shape:", left_vector.shape)

    right_vector: torch.Tensor = compute_v(
        model,
        tok,
        request,
        hparams,
        hparams.layer,
        left_vector,
        context_templates,
    )
    print("Right vector shape:", right_vector.shape)

    return left_vector, right_vector
