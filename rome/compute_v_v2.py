from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .hparams import ROMEHyperParams
from .preserving import TextRomePreserving, TokenizedRomePreserving
from .rewriting import TextRomeRewriting, TokenizedRomeRewriting
from .utils import nethook
from .utils.batching import stack_with_padding
from .utils.hooks import StopForward, forward_output_hook
from .utils.nethook import get_module


def compute_v(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    rewriting: TextRomeRewriting,
    preservings: Iterable[TextRomePreserving] | None,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    prefixes: list[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    if preservings is None:
        preservings = [TextRomePreserving(
            prompt=f"{rewriting.subject} is a ",
            subject_head=0,
            subject_tail=len(rewriting.subject)
        )]

    prefixes_tokenized = [np.asarray(tokenizer.encode(prefix), dtype=np.int64) for prefix in prefixes]
    rewriting_tokenized = TokenizedRomeRewriting.from_text_rewriting(rewriting, tokenizer)
    preservings_tokenized = [TokenizedRomePreserving.from_text_preserving(preserving, tokenizer) for preserving in
                             preservings]

    rewritings_inputs = [
        np.concatenate([
            prefix_tokenized,
            rewriting_tokenized.prompt,
            rewriting_tokenized.target])
        for prefix_tokenized in prefixes_tokenized]
    rewritings_subject_token_index = [
        (len(prefix_tokenized) + rewriting_tokenized.subject_tail - 1)
        for prefix_tokenized in prefixes_tokenized]
    rewritings_target_tokens_prob_coo = [
        (i, len(prefix_tokenized) + len(rewriting_tokenized.prompt) - 1 + j, target_token)
        for j, target_token in enumerate(rewriting_tokenized.target)
        for i, prefix_tokenized in enumerate(prefixes_tokenized)]

    preservings_inputs = [
        preserving.prompt
        for preserving in preservings_tokenized]
    preservings_subject_token_index = [
        (preserving.subject_tail - 1)
        for preserving in preservings_tokenized]

    all_in_tokens = stack_with_padding([
        *rewritings_inputs,
        *preservings_inputs,
    ], tokenizer.pad_token_id)
    all_in_tokens = torch.asarray(all_in_tokens, dtype=torch.int64, device=model.device)
    all_in_subject_token_index = [
        *rewritings_subject_token_index,
        *preservings_subject_token_index]

    # compute k, v without prefixes
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    def hook_func(_, inputs: tuple[torch.Tensor], output: torch.Tensor):
        input, = inputs
        nonlocal k, v
        k = input[0, rewriting_tokenized.subject_tail - 1].clone()
        v = output[0, rewriting_tokenized.subject_tail - 1].clone()
        raise StopForward()

    with torch.no_grad(), forward_output_hook(get_module(model, hparams.rewrite_module_tmp.format(layer)), hook_func):
        model(torch.asarray(rewriting_tokenized.prompt, dtype=torch.int64, device=model.device))

    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)

    preservings_log_probs_init: torch.Tensor | None = None
    v_norm = torch.norm(v)

    delta = torch.zeros(v.shape, dtype=v.dtype, device=v.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(_, __, cur_out: torch.Tensor) -> torch.Tensor:
        for i, idx in enumerate(all_in_subject_token_index):
            cur_out[i, idx, :] += delta
        return cur_out

    # Execute optimization
    nethook.set_requires_grad(False, model)
    edit_module = get_module(model, hparams.rewrite_module_tmp.format(layer))
    for it in range(hparams.v_num_grad_steps):
        # Forward propagation
        with forward_output_hook(edit_module, edit_output_fn):
            all_out_tokens_logits = model(all_in_tokens).logits
        rewritings_out_tokens_logits = all_out_tokens_logits[:len(rewritings_inputs)]
        preservings_out_tokens_logits = all_out_tokens_logits[len(rewritings_inputs):]

        # Compute distribution for KL divergence
        coo_i, coo_j = range(len(preservings_inputs)), preservings_subject_token_index
        preservings_logits = preservings_out_tokens_logits[coo_i, coo_j, :]
        preservings_log_probs = torch.nn.functional.log_softmax(preservings_logits, dim=1)
        if preservings_log_probs_init is None:
            preservings_log_probs_init = preservings_log_probs.detach().clone()
        preserving_loss = torch.nn.functional.kl_div(
            preservings_log_probs_init, preservings_log_probs,
            log_target=True, reduction="batchmean")

        # Compute loss on rewriting targets
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
        opt.step()
        opt.zero_grad()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * v_norm
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = v + delta

    # Solving the linear system to compute the right vector
    right_vector = (target - v) / torch.dot(k, left_vector)
    print(f"Delta norm: {(target - v).norm().item()}")
    print(f"Change in target norm: {v_norm.item()} to {target.norm().item()} => {(target.norm() - v_norm).item()}")
    print(f"Division Factor: {torch.dot(k, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector