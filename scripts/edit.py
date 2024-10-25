import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

project_dir_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_dir_path)

from scripts.utils import caching_torch_tensor, compute_c_inv, print_v_delta_metrics, generate_text
from zkl_rome import ComputeCHparams, ComputeVDeltaHparams, GeneratePrefixesHparams, TextRewriting, apply_left_right, \
    compute_left_right, generate_prefixes, generate_preservings_by_default

# config

device = "cuda"

model_name = "gpt2-medium"
module_name = "transformer.h.8.mlp.c_proj"

c_inv_cache_path = os.path.join(project_dir_path, f"caches/{model_name}/{module_name}/c_inv.pt")

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
    total_tokens_num=int(1e6),
    batch_samples_num=4,
    context_tokens_num=256)

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
module = model.get_submodule(module_name)
rewriting_tokenized = rewriting.tokenize(tokenizer)
prefixes_tokenized = generate_prefixes(model, tokenizer, generate_prefixes_hparams)
preservings_tokenized = generate_preservings_by_default(tokenizer, rewriting_tokenized)

compute_c_inv = caching_torch_tensor(c_inv_cache_path)(compute_c_inv)
c_inv = compute_c_inv(compute_c_hparams, model, module, tokenizer)

(left, right) = compute_left_right(
    model=model,
    module=module,
    prefixes=prefixes_tokenized,
    rewriting=rewriting_tokenized,
    preservings=preservings_tokenized,
    c_inv=c_inv,
    compute_v_delta_hparams=compute_v_delta_hparams,
    compute_v_delta_callback=print_v_delta_metrics)
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
