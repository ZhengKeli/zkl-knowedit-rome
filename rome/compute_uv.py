import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_u import compute_u
from .compute_v import compute_v
from .hparams import ROMEHyperParams
from .prefixes import iter_random_prefixes
from .rewriting import TextRomeRewriting


def execute_rome(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    rewriting: TextRomeRewriting,
    hparams: ROMEHyperParams,
    stats_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # prefixes
    prefixes = list(iter_random_prefixes(model, tokenizer, hparams.context_template_length_params))

    u = compute_u(
        hparams,
        model,
        tokenizer,
        prefixes,
        rewriting,
        stats_dir)

    v = compute_v(
        model,
        tokenizer,
        rewriting,
        None,
        hparams,
        hparams.layer,
        u,
        prefixes)

    return u, v
