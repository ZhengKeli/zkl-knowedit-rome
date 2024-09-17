import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import repr_tools
from .hparams import ROMEHyperParams
from .request import TextRomeRequest


def compute_k(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: TextRomeRequest,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: list[str],
) -> torch.Tensor:
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in")

    cur_repr = repr_tools.get_reprs_at_word_tokens(
        context_templates=[templ.format(request.prompt_template) for templ in context_templates],
        words=[request.subject for _ in range(len(context_templates))],
        **word_repr_args,
    ).mean(0)

    return cur_repr
