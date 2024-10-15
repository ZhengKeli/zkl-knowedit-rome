import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

project_dir_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_dir_path)

from zkl_rome import RomeComputeCHParams, RomeComputeVDeltaHparams, TextRomeRewriting, TokenizedRomeRewriting, \
    compute_c, compute_left_right, make_default_prefixes, make_default_preservings

# config

device = "cuda"

model_name = "gpt2-medium"
module_name = "transformer.h.8.mlp.c_proj"

rewriting = TextRomeRewriting(
    prompt="Steve Jobs is the founder of",
    subject="Steve Jobs",
    target=" Microsoft")

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

c_org_path = os.path.join(project_dir_path, "../original-rome/c.npy")
c_inv_org_path = os.path.join(project_dir_path, "../original-rome/c_inv.npy")
left_org_path = os.path.join(project_dir_path, "../original-rome/left.npy")
right_org_path = os.path.join(project_dir_path, "../original-rome/right.npy")
w_delta_org_path = os.path.join(project_dir_path, "../original-rome/w_delta.npy")


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


def compare(a1: torch.Tensor, a2: torch.Tensor):
    a1 = torch.reshape(a1, [-1])
    a2 = torch.reshape(a2, [-1])
    return torch.nn.functional.cosine_similarity(a1, a2, dim=0)


# execution

print(f"Loading Model and Tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device=device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

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
w_delta = torch.outer(left, right)

# comparing

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
