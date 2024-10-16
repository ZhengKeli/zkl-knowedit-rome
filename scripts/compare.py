import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

project_dir_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_dir_path)

from scripts.utils import iter_samples_for_compute_c, load_dataset_for_compute_c, print_v_delta_metrics
from zkl_rome import ComputeCHparams, ComputeVDeltaHparams, TextRewriting, compute_c, compute_left_right, \
    make_default_prefixes, make_default_preservings

# config

device = "cuda"

model_name = "gpt2-medium"
module_name = "transformer.h.8.mlp.c_proj"

rewriting = TextRewriting(
    prompt="Steve Jobs is the founder of",
    subject="Steve Jobs",
    target=" Microsoft")

compute_c_hparams = ComputeCHparams(
    total_tokens_num=100000,
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

c_org_path = os.path.join(project_dir_path, "../original-rome/c.npy")
c_inv_org_path = os.path.join(project_dir_path, "../original-rome/c_inv.npy")
left_org_path = os.path.join(project_dir_path, "../original-rome/left.npy")
right_org_path = os.path.join(project_dir_path, "../original-rome/right.npy")
w_delta_org_path = os.path.join(project_dir_path, "../original-rome/w_delta.npy")


# utils

def compare(a1: torch.Tensor, a2: torch.Tensor):
    a1 = torch.reshape(a1, [-1])
    a2 = torch.reshape(a2, [-1])
    return torch.nn.functional.cosine_similarity(a1, a2, dim=0)


# execution

print(f"Loading Model and Tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading Dataset (for compute c)")
dataset = load_dataset_for_compute_c()

print(f"Applying ROME to model")
module = model.get_submodule(module_name)
rewriting_tokenized = rewriting.tokenize(tokenizer)
prefixes_tokenized = make_default_prefixes(model, tokenizer)
preservings_tokenized = make_default_preservings(tokenizer, rewriting_tokenized)

c = compute_c(
    compute_c_hparams,
    model, module,
    iter_samples_for_compute_c(dataset, tokenizer))
c_inv = torch.inverse(c)

(left, right) = compute_left_right(
    compute_v_delta_hparams,
    model, module,
    rewriting_tokenized,
    prefixes_tokenized,
    preservings_tokenized,
    c_inv,
    compute_v_delta_callback=print_v_delta_metrics)
w_delta = torch.outer(left, right)

print(f"Comparing results")

c_org = np.load(c_org_path)
c_org = torch.from_numpy(c_org).to(c)
c_sim = compare(c, c_org).item()
print(f"{c_sim=}")

c_inv_org = np.load(c_inv_org_path)
c_inv_org = torch.from_numpy(c_inv_org).to(c_inv)
c_inv_sim = compare(c_inv, c_inv_org).item()
print(f"{c_inv_sim=}")

left_org = np.load(left_org_path)
left_org = torch.from_numpy(left_org).to(left)
left_sim = compare(left, left_org).item()
print(f"{left_sim=}")

right_org = np.load(right_org_path)
right_org = torch.from_numpy(right_org).to(right)
right_sim = compare(right, right_org).item()
print(f"{right_sim=}")

w_delta_org = np.load(w_delta_org_path)
w_delta_org = torch.from_numpy(w_delta_org).to(w_delta)
w_delta_sim = compare(w_delta, w_delta_org).item()
print(f"{w_delta_sim=}")
