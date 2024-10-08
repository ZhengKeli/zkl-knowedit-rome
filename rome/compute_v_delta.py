from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .hparams import ROMEHyperParams
from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting
from .utils import nethook
from .utils.batching import stack_with_padding
from .utils.hooks import forward_output_hook
from .utils.nethook import get_module


def compute_v_delta(
    hparams: ROMEHyperParams,
    model: PreTrainedModel,
    layer: int,
    prefixes: Iterable[np.ndarray],
    rewriting: TokenizedRomeRewriting,
    preservings: Iterable[TokenizedRomePreserving],
    v: torch.Tensor,
) -> torch.Tensor:
    prefixes = tuple(prefixes)

    rewritings_inputs = [
        np.concatenate([
            prefix,
            rewriting.prompt,
            rewriting.target])
        for prefix in prefixes]
    rewritings_subject_token_index = [
        (len(prefix) + rewriting.subject_tail - 1)
        for prefix in prefixes]
    rewritings_target_tokens_prob_coo = [
        (i, len(prefix_tokenized) + len(rewriting.prompt) - 1 + j, target_token)
        for j, target_token in enumerate(rewriting.target)
        for i, prefix_tokenized in enumerate(prefixes)]

    preservings_inputs = [
        preserving.prompt
        for preserving in preservings]
    preservings_subject_token_index = [
        (preserving.subject_tail - 1)
        for preserving in preservings]

    all_in_tokens = torch.asarray(stack_with_padding([
        *rewritings_inputs,
        *preservings_inputs,
    ], 0), dtype=torch.int64, device=model.device)
    all_in_subject_token_index = [
        *rewritings_subject_token_index,
        *preservings_subject_token_index]

    delta = torch.zeros_like(v, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=hparams.v_lr)

    v_norm = torch.norm(v)
    preservings_log_probs_init: torch.Tensor | None = None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(_, __, output: torch.Tensor) -> torch.Tensor:
        for i, idx in enumerate(all_in_subject_token_index):
            output[i, idx, :] += delta
        return output

    # Execute optimization
    nethook.set_requires_grad(False, model)
    edit_module = get_module(model, hparams.rewrite_module_tmp.format(layer))
    for it in range(hparams.v_num_grad_steps):
        # Forward propagation
        with forward_output_hook(edit_module, edit_output_fn):
            all_out_tokens_logits = model(all_in_tokens).logits

        # Compute distribution for KL divergence
        preservings_out_tokens_logits = all_out_tokens_logits[len(rewritings_inputs):]
        coo_i, coo_j = range(len(preservings_inputs)), preservings_subject_token_index
        preservings_logits = preservings_out_tokens_logits[coo_i, coo_j, :]
        preservings_log_probs = torch.log_softmax(preservings_logits, dim=-1)
        if preservings_log_probs_init is None:
            preservings_log_probs_init = preservings_log_probs.detach().clone()
        preserving_loss = torch.nn.functional.kl_div(
            preservings_log_probs_init, preservings_log_probs,
            log_target=True, reduction="batchmean")

        # Compute loss on rewriting targets
        rewritings_out_tokens_logits = all_out_tokens_logits[:len(rewritings_inputs)]
        rewritings_out_tokens_log_probs = torch.log_softmax(rewritings_out_tokens_logits, dim=-1)
        coo_i, coo_j, coo_k = zip(*rewritings_target_tokens_prob_coo)
        rewritings_out_tokens_log_prob = rewritings_out_tokens_log_probs[coo_i, coo_j, coo_k]
        rewriting_loss = -torch.mean(rewritings_out_tokens_log_prob)

        # Compute loss on regularization
        regularization_loss = hparams.v_weight_decay * (torch.norm(delta) / v_norm ** 2)
        # regularization_loss = hparams.v_weight_decay * torch.norm(delta) ** 2

        loss = rewriting_loss + hparams.kl_factor * preserving_loss + regularization_loss

        print(", ".join([
            f"loss={loss.item():.3f}",
            f"rewriting_loss={rewriting_loss.item():.3f}",
            f"preserving_loss={preserving_loss.item():.3f}",
            f"regularization_loss={regularization_loss.item():.3f}",
            f"prob={rewritings_out_tokens_log_prob.exp().mean().item():.3f}"]))

        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * v_norm
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    return delta