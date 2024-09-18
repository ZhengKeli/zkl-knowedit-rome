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
    prefixes: list[str],
) -> torch.Tensor:
    return repr_tools.get_reprs_at_word_tokens(
        model=model,
        tok=tokenizer,
        module_template=hparams.rewrite_module_tmp,
        layer=layer,
        inputs=[templ + request.prompt_template for templ in prefixes],
        words=[request.subject for _ in range(len(prefixes))],
        track="in").mean(0)
