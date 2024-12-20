import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_dir_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_dir_path)

from scripts.utils import compute_cosine_similarity
from zkl_rome import ComputeCHparams, ComputeVDeltaHparams, TextRewriting, TqdmComputeCCallback, \
    TqdmComputeVDeltaCallback, WikipediaComputeCSamples, compute_c, compute_left_right, generate_prefixes_by_default, \
    generate_preservings_by_default

# config

model_name = "gpt2-medium"
module_name = "transformer.h.8.mlp.c_proj"

device = "cuda" if torch.cuda.is_available() else "cpu"

rewriting = TextRewriting(
    prompt="Steve Jobs is the founder of",
    subject="Steve Jobs",
    target=" Microsoft")

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

org_path = os.path.join(project_dir_path, "compare", model_name, module_name)
c_org_path = os.path.join(org_path, "c.npy")
c_inv_org_path = os.path.join(org_path, "c_inv.npy")
left_org_path = os.path.join(org_path, "left.npy")
right_org_path = os.path.join(org_path, "right.npy")
w_delta_org_path = os.path.join(org_path, "w_delta.npy")

# execution

print(f"Loading Model and Tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print(f"Computing ROME intermediates")
module = model.get_submodule(module_name)
rewriting = rewriting.tokenize(tokenizer)
prefixes = generate_prefixes_by_default(model, tokenizer)
preservings = generate_preservings_by_default(tokenizer, rewriting)

c = compute_c(
    model=model,
    module=module,
    samples=WikipediaComputeCSamples(tokenizer=tokenizer),
    hparams=compute_c_hparams,
    callback=TqdmComputeCCallback())
c_inv = torch.inverse(c)

(left, right) = compute_left_right(
    model=model,
    module=module,
    prefixes=prefixes,
    rewriting=rewriting,
    preservings=preservings,
    c_inv=c_inv,
    compute_v_delta_hparams=compute_v_delta_hparams,
    compute_v_delta_callback=TqdmComputeVDeltaCallback())

w_delta = torch.outer(left, right)

print(f"Comparing results")

c_org = np.load(c_org_path)
c_org = torch.from_numpy(c_org).to(c)
c_sim = compute_cosine_similarity(c, c_org).item()
print(f"{c_sim=}")

c_inv_org = np.load(c_inv_org_path)
c_inv_org = torch.from_numpy(c_inv_org).to(c_inv)
c_inv_sim = compute_cosine_similarity(c_inv, c_inv_org).item()
print(f"{c_inv_sim=}")

left_org = np.load(left_org_path)
left_org = torch.from_numpy(left_org).to(left)
left_sim = compute_cosine_similarity(left, left_org).item()
print(f"{left_sim=}")

right_org = np.load(right_org_path)
right_org = torch.from_numpy(right_org).to(right)
right_sim = compute_cosine_similarity(right, right_org).item()
print(f"{right_sim=}")

w_delta_org = np.load(w_delta_org_path)
w_delta_org = torch.from_numpy(w_delta_org).to(w_delta)
w_delta_sim = compute_cosine_similarity(w_delta, w_delta_org).item()
print(f"{w_delta_sim=}")
