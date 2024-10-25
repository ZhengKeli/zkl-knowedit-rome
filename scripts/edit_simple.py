import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

project_dir_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_dir_path)

from scripts.utils import generate_text
from zkl_rome import ComputeVDeltaHparams, TextRewriting, rome

# config

device = "cuda"

model_name = "gpt2-medium"
module_name = "transformer.h.8.mlp.c_proj"

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

cache_c_inv_file_path = os.path.join(project_dir_path, f"caches/{model_name}/{module_name}/c_inv.pt")

compute_v_delta_hparams = ComputeVDeltaHparams(
    learning_rate=5e-1,
    stopping_steps_num=20,
    stopping_loss_threshold=5e-2,
    rewriting_loss_k=1.0,
    preserving_loss_k=0.0625,
    regularization_loss_k=0.5,
    regularization_constraint_factor=3.0)

# execution

print(f"Loading Model and Tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Generating pre-update text")
pre_update_text = generate_text(model, tokenizer, inspecting_prompts)

print(f"Applying ROME to model")
rome(
    model=model,
    tokenizer=tokenizer,
    module_name=module_name,
    rewriting=rewriting,
    cache_c_inv_file_path=cache_c_inv_file_path,
    compute_v_delta_hparams=compute_v_delta_hparams)

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
