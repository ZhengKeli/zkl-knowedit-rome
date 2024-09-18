import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .compute_u import compute_u
from .compute_v import compute_v
from .hparams import ROMEHyperParams
from .prefixes import iter_random_prefixes
from .rewriting import TextRomeRewriting


def execute_rome(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    rewriting: TextRomeRewriting,
    hparams: ROMEHyperParams,
    stats_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # prefixes
    prefixes = list(iter_random_prefixes(model, tok, hparams.context_template_length_params))

    # Compute rank-1 update matrix
    left_vector: torch.Tensor = compute_u(
        model,
        tok,
        rewriting,
        hparams,
        hparams.layer,
        prefixes,
        stats_dir,
    )
    print("Left vector shape:", left_vector.shape)

    right_vector: torch.Tensor = compute_v(
        model,
        tok,
        rewriting,
        hparams,
        hparams.layer,
        left_vector,
        prefixes,
    )
    print("Right vector shape:", right_vector.shape)

    return left_vector, right_vector
