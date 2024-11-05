import abc
from dataclasses import dataclass
from itertools import count
from typing import Callable, Iterable

import numpy as np
import torch
from transformers import PreTrainedModel

from .batching import stack_with_aligning
from .hooks import forward_output_hook, no_grad_from
from .preserving import TokenizedPreserving
from .rewriting import TokenizedRewriting


@dataclass(kw_only=True)
class ComputeVDeltaHparams:
    learning_rate: float

    rewriting_loss_k: float = 1.0
    preserving_loss_k: float = 1.0
    regularization_loss_k: float = 0.0
    regularization_constraint_factor: float | None = None

    stopping_steps_num: int | None = 100
    stopping_loss_threshold: float | None = 5e-2


@dataclass(kw_only=True)
class ComputeVDeltaMetrics:
    processed_steps_num: int

    rewriting_acc: torch.Tensor
    rewriting_loss: torch.Tensor
    preserving_loss: torch.Tensor
    regularization_loss: torch.Tensor
    loss: torch.Tensor


class ComputeVDeltaCallback(abc.ABC):
    @abc.abstractmethod
    def on_start(self, hparams: ComputeVDeltaHparams):
        pass

    @abc.abstractmethod
    def on_step(self, metrics: ComputeVDeltaMetrics):
        pass

    @abc.abstractmethod
    def on_stop(self, metrics: ComputeVDeltaMetrics):
        pass


def compute_v_delta(*,
    model: PreTrainedModel,
    module: torch.nn.Module,
    prefixes: Iterable[np.ndarray],
    rewriting: TokenizedRewriting,
    preservings: Iterable[TokenizedPreserving],
    v: torch.Tensor,
    hparams: ComputeVDeltaHparams,
    callback: ComputeVDeltaCallback | None = None,
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

    all_in_tokens = torch.asarray(stack_with_aligning([
        *rewritings_inputs,
        *preservings_inputs,
    ], pad=0), dtype=torch.int64, device=model.device)
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

    # Call callback
    if callback is not None:
        callback.on_start(hparams)

    # Execute optimization
    metrics = None
    for step_i in count():
        # Forward propagation
        with no_grad_from(*model.parameters()), forward_output_hook(module, edit_output_fn):
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
        rewriting_acc = rewritings_out_tokens_log_prob.exp()
        rewriting_loss = -torch.mean(rewritings_out_tokens_log_prob)

        # Compute loss on regularization
        regularization_loss = torch.norm(v_delta) / v_norm ** 2
        # regularization_loss = torch.norm(v_delta) ** 2

        # Compute total loss
        loss = (hparams.rewriting_loss_k * rewriting_loss +
                hparams.preserving_loss_k * preserving_loss +
                hparams.regularization_loss_k * regularization_loss)

        # Call callback
        if callback is not None:
            metrics = ComputeVDeltaMetrics(
                processed_steps_num=step_i,
                rewriting_acc=rewriting_acc.detach(),
                rewriting_loss=rewriting_loss.detach(),
                preserving_loss=preserving_loss.detach(),
                regularization_loss=regularization_loss.detach(),
                loss=loss.detach())
            callback.on_step(metrics)

        # Stop by steps num
        if hparams.stopping_steps_num is not None:
            if step_i >= hparams.stopping_steps_num:
                break

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

    # Call callback
    if callback is not None:
        assert isinstance(metrics, ComputeVDeltaMetrics)
        callback.on_stop(metrics)

    return v_delta
