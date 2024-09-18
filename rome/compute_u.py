import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_c import compute_c_inv
from .compute_k import compute_k
from .hparams import ROMEHyperParams
from .rewriting import TextRomeRewriting


def compute_u(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    rewriting: TextRomeRewriting,
    hparams: ROMEHyperParams,
    layer: int,
    prefixes: list[str],
    stats_dir: str,
) -> torch.Tensor:
    u = compute_k(
        model,
        tok,
        rewriting,
        hparams,
        layer,
        prefixes)

    # Apply inverse second moment adjustment
    if hparams.mom2_adjustment:
        c_inv = compute_c_inv(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            stats_dir)
        c_inv = c_inv.to(u)
        u = c_inv @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
