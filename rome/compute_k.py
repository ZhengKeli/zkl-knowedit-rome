import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import repr_tools
from .hparams import ROMEHyperParams
from .request import TextRomeRequest


def compute_k(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    request: TextRomeRequest,
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: list[str],
) -> torch.Tensor:
    return repr_tools.get_reprs_at_word_tokens(
        model=model,
        tok=tokenizer,
        module_template=hparams.rewrite_module_tmp,
        layer=layer,
        context_templates=[templ.format(request.prompt_template) for templ in context_templates],
        words=[request.subject for _ in range(len(context_templates))],
        track="in").mean(0)
