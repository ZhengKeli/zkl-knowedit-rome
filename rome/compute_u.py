import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import repr_tools
from .compute_c import get_inv_cov
from .hparams import ROMEHyperParams


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: dict,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: list[str],
    stats_dir: str,
) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )
    word = request["subject"]
    print(f"Selected u projection object {word}")
    cur_repr = repr_tools.get_reprs_at_word_tokens(
        context_templates=[
            templ.format(request["prompt"]) for templ in context_templates
        ],
        words=[word for _ in range(len(context_templates))],
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
