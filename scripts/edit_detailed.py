import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_dir_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_dir_path)

from scripts.utils import generate_text, iter_compute_c_samples_from_wikipedia
from zkl_rome import ComputeCHparams, ComputeVDeltaHparams, GeneratePrefixesHparams, TextRewriting, \
    TqdmComputeCCallback, TqdmComputeVDeltaCallback, apply_left_right, compute_left_right, generate_prefixes, \
    generate_preservings_by_default, load_or_compute_c_inv

# config

model_name = "gpt2-medium"
module_name = "transformer.h.8.mlp.c_proj"

device = "cuda" if torch.cuda.is_available() else "cpu"

rewriting = TextRewriting(
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

generate_prefixes_hparams = GeneratePrefixesHparams(
    seperator=". ",
    num_tokens=10,
    num_sequences=20)

compute_c_hparams = ComputeCHparams(
    batch_samples_num=4,
    context_tokens_num=256,
    stopping_tokens_num=int(1e6))

compute_v_delta_hparams = ComputeVDeltaHparams(
    learning_rate=5e-1,
    stopping_steps_num=20,
    stopping_loss_threshold=5e-2,
    rewriting_loss_k=1.0,
    preserving_loss_k=0.0625,
    regularization_loss_k=0.5,
    regularization_constraint_factor=3.0)

cache_c_inv_file_path = os.path.join(project_dir_path, f"caches/{model_name}/{module_name}/c_inv.pt")

# execution

print(f"Loading Model and Tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Generating pre-update text")
pre_update_text = generate_text(model, tokenizer, inspecting_prompts)

print(f"Applying ROME to model")
module = model.get_submodule(module_name)
rewriting = rewriting.tokenize(tokenizer)
prefixes = generate_prefixes(model, tokenizer, generate_prefixes_hparams)
preservings = generate_preservings_by_default(tokenizer, rewriting)

c_inv = load_or_compute_c_inv(
    model=model,
    module=module,
    compute_c_samples=iter_compute_c_samples_from_wikipedia(tokenizer),
    compute_c_hparams=compute_c_hparams,
    compute_c_callback=TqdmComputeCCallback(),
    cache_c_inv_file_path=cache_c_inv_file_path)

(left, right) = compute_left_right(
    model=model,
    module=module,
    prefixes=prefixes,
    rewriting=rewriting,
    preservings=preservings,
    c_inv=c_inv,
    compute_v_delta_hparams=compute_v_delta_hparams,
    compute_v_delta_callback=TqdmComputeVDeltaCallback())

apply_left_right(module, left, right)

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
