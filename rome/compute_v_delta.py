from dataclasses import dataclass
from itertools import count
from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .preserving import TokenizedRomePreserving
from .rewriting import TokenizedRomeRewriting
from .utils import nethook
from .utils.batching import stack_with_padding
from .utils.hooks import forward_output_hook


@dataclass(kw_only=True)
class RomeComputeVDeltaHparams:
    learning_rate: float

    stopping_steps_num: int | None = 100
    stopping_loss_threshold: float | None = 5e-2

    rewriting_loss_k: float = 1.0
    preserving_loss_k: float = 1.0
    regularization_loss_k: float = 0.0
    regularization_constraint_factor: float | None = None


def compute_v_delta(
    hparams: RomeComputeVDeltaHparams,
    model: PreTrainedModel,
    module: torch.nn.Module,
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

    v_delta = torch.zeros_like(v, requires_grad=True)
    optimizer = torch.optim.Adam([v_delta], lr=hparams.learning_rate)

    v_norm = torch.norm(v)
    preservings_log_probs_init: torch.Tensor | None = None

    # Add "v_delta" at the appropriate part of the computation
    def edit_output_fn(_, __, output: torch.Tensor) -> torch.Tensor:
        for i, idx in enumerate(all_in_subject_token_index):
            output[i, idx, :] += v_delta
        return output

    # Execute optimization
    nethook.set_requires_grad(False, model)
    for step_i in count():
        # Stop by steps num
        if hparams.stopping_steps_num is not None:
            if step_i >= hparams.stopping_steps_num:
                break

        # Forward propagation
        with forward_output_hook(module, edit_output_fn):
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
        regularization_loss = torch.norm(v_delta) / v_norm ** 2
        # regularization_loss = torch.norm(v_delta) ** 2

        # Compute total loss
        loss = (hparams.rewriting_loss_k * rewriting_loss +
                hparams.preserving_loss_k * preserving_loss +
                hparams.regularization_loss_k * regularization_loss)

        print(", ".join([
            f"loss={loss.item():.3f}",
            f"rewriting_loss={rewriting_loss.item():.3f}",
            f"preserving_loss={preserving_loss.item():.3f}",
            f"regularization_loss={regularization_loss.item():.3f}",
            f"prob={rewritings_out_tokens_log_prob.exp().mean().item():.3f}"]))

        # Stop by loss threshold
        if hparams.stopping_loss_threshold is not None:
            if loss < hparams.stopping_loss_threshold:
                break

        # Back-propagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Constrain within L2 ball
        if hparams.regularization_constraint_factor is not None:
            max_norm = hparams.regularization_constraint_factor * v_norm
            if v_delta.norm() > max_norm:
                with torch.no_grad():
                    v_delta[...] = v_delta * max_norm / v_delta.norm()

    return v_delta
