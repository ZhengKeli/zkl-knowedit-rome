import os
import sys
from typing import Iterable

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, pipeline

project_dir_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_dir_path)

from zkl_rome import RomeComputeCHParams, RomeComputeVDeltaHparams, TextRomeRewriting, TokenizedRomeRewriting, \
    apply_left_right_to_module, compute_c, compute_left_right, make_default_prefixes, make_default_preservings

# config

device = "cuda"

model_name = "gpt2-medium"
module_name = "transformer.h.8.mlp.c_proj"

rewriting = TextRomeRewriting(
    prompt="Steve Jobs is the founder of",
    subject="Steve Jobs",
    target=" Microsoft")

inspecting_prompts = [
    "My favorite Steve Jobs product is",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
    "Steve Jobs was the founder of",
]

compute_c_hparams = RomeComputeCHParams(
    total_tokens_num=100000,
    batch_samples_num=4,
    context_tokens_num=256)

compute_v_delta_hparams = RomeComputeVDeltaHparams(
    learning_rate=5e-1,
    stopping_steps_num=20,
    stopping_loss_threshold=5e-2,
    rewriting_loss_k=1.0,
    preserving_loss_k=0.0625,
    regularization_loss_k=0.5,
    regularization_constraint_factor=3.0)


# utils

def iter_tokenized_texts_from_dataset(tokenizer: PreTrainedTokenizer):
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        trust_remote_code=True,
        streaming=True)

    for sample in dataset:
        sample = sample["text"]
        sample = tokenizer.encode(sample)
        sample = np.asarray(sample, dtype=np.int64)
        yield sample


def generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompts: Iterable[str]):
    pipe = pipeline("text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        num_return_sequences=1,
        return_full_text=True,
        max_new_tokens=64)
    return tuple(pipe(prompt)[0]['generated_text'] for prompt in prompts)


# execution

print(f"Loading Model and Tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Generating pre-update text")
pre_update_text = generate_text(model, tokenizer, inspecting_prompts)

print(f"Applying ROME to model")
module = model.get_submodule(module_name)
rewriting_tokenized = TokenizedRomeRewriting.from_text_rewriting(rewriting, tokenizer)
prefixes_tokenized = make_default_prefixes(model, tokenizer)
preservings_tokenized = make_default_preservings(tokenizer, rewriting_tokenized)

c = compute_c(
    compute_c_hparams,
    model, module,
    iter_tokenized_texts_from_dataset(tokenizer))
c_inv = torch.inverse(c)

(left, right) = compute_left_right(
    compute_v_delta_hparams,
    model, module,
    rewriting_tokenized,
    prefixes_tokenized,
    preservings_tokenized,
    c_inv)
apply_left_right_to_module(module, left, right)

print("Generating post-update text")
post_update_text = generate_text(model, tokenizer, inspecting_prompts)

print("Summarizing differences")
for i, (prompt, pre, post) in enumerate(zip(inspecting_prompts, pre_update_text, post_update_text)):
    if i > 0:
        print("".join(["-" for _ in range(10)]))

    prompt_str = "[Prompt]:"
    pre_str = f"[Pre]:"
    post_str = f"[Post]:"
    pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

    for s, t in zip([prompt_str, pre_str, post_str], [prompt, pre, post]):
        print(s.ljust(pad_to), t)
