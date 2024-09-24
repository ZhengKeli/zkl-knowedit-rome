from typing import Iterable

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from . import repr_tools
from .hparams import ROMEHyperParams
from .preserving import TextRomePreserving, TokenizedRomePreserving
from .rewriting import TextRomeRewriting, TokenizedRomeRewriting
from .utils import nethook
from .utils.batching import stack_with_padding
from .utils.hooks import forward_output_hook
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

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    target_init, preservings_log_probs_init = None, None

    # Optimizer
    delta: torch.Tensor | None = None
    opt: torch.optim.Optimizer | None = None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(_, __, cur_out: torch.Tensor) -> torch.Tensor:
        nonlocal target_init, delta, opt

        # Store initial value of the vector of interest
        if target_init is None:
            print("Recording initial value of v*")
            # Initial value is recorded for the clean sentence
            target_init = cur_out[0, all_in_subject_token_index[0]].detach().clone()

        if delta is None:
            delta = torch.zeros(cur_out.shape[-1:], dtype=cur_out.dtype, device=cur_out.device, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=hparams.v_lr)

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
        regularization_loss = hparams.v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
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
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tokenizer,
        layer,
        input=rewriting.prompt_template,
        word=rewriting.subject,
        module_template=hparams.rewrite_module_tmp,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}")
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: PreTrainedModel,
    tok: PreTrainedTokenizer,
    layer: int,
    input: str,
    word: str,
    module_template: str,
) -> tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    l_input, l_output = repr_tools.get_reprs_at_word_tokens(
        track="both",
        inputs=[input],
        words=[word],
        **word_repr_args,
    )

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()
