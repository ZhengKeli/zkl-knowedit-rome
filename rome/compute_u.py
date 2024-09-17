import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import repr_tools
from .compute_c import get_inv_cov
from .hparams import ROMEHyperParams
from .request import TextRomeRequest


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: TextRomeRequest,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: list[str],
    stats_dir: str,
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    subject = request.subject
    prompt_temp = request.prompt[:request.subject_head] + "{}" + request.prompt[request.subject_tail:]

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )

    cur_repr = repr_tools.get_reprs_at_word_tokens(
        context_templates=[templ.format(prompt_temp) for templ in context_templates],
        words=[subject for _ in range(len(context_templates))],
        subtoken="last",
        **word_repr_args,
    ).mean(0)

    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            stats_dir
        ) @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
