import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_u import compute_u
from .compute_v import compute_v
from .hparams import ROMEHyperParams
from .prefixes import get_context_templates
from .request import TextRomeRequest


def execute_rome(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    request: TextRomeRequest,
    hparams: ROMEHyperParams,
    stats_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    request = {
        'prompt': request.prompt[:request.subject_head] + "{}" + request.prompt[request.subject_tail:],
        'subject': request.subject,
        'target_new': {"str": request.target},
    }

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
