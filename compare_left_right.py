import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from zkl_serialization import load_and_parse_json

from rome import RomeHparams, TextRomeRewriting, TokenizedRomeRewriting, compute_c_inv, compute_left_right, make_default_prefixes, make_default_preservings

model_name = "gpt2-medium"
hparams_file_path = os.path.join("hparams/ROME/gpt2-medium.json")
stats_dir = "data/stats"

rewriting = TextRomeRewriting(
    prompt="Steve Jobs is the founder of",
    subject="Steve Jobs",
    target=" Microsoft")

generation_prompts = [
    "My favorite Steve Jobs product is",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
    "Steve Jobs was the founder of",
]

print(f"Loading Model and Tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print(f"Retrieving hyperparameters")
print("Loading from", hparams_file_path)
hparams = load_and_parse_json(hparams_file_path, RomeHparams)
print(hparams)

print(f"Applying ROME to model")

rewriting_tokenized = TokenizedRomeRewriting.from_text_rewriting(rewriting, tokenizer)
prefixes_tokenized = make_default_prefixes(model, tokenizer)
preservings_tokenized = make_default_preservings(tokenizer, rewriting_tokenized)

c_inv = compute_c_inv(
    model,
    tokenizer,
    hparams.rewrite_module_name,
    hparams.mom2_dataset,
    hparams.mom2_n_samples,
    hparams.mom2_dtype,
    stats_dir
) if hparams.mom2_adjustment else None

module = model.get_submodule(hparams.rewrite_module_name)

(left, right) = compute_left_right(model, module, rewriting_tokenized, prefixes_tokenized, preservings_tokenized, hparams.v_delta, c_inv)
delta_weight = torch.outer(left, right)

left_original = np.load("../zkl-knowedit-rome-original/left.npy")
left_original = torch.from_numpy(left_original).to(left)

right_original = np.load("../zkl-knowedit-rome-original/right.npy")
right_original = torch.from_numpy(right_original).to(right)

delta_weight_original = np.load("../zkl-knowedit-rome-original/delta_weight.npy")
delta_weight_original = torch.from_numpy(delta_weight_original).to(delta_weight)


def compare(a1: torch.Tensor, a2: torch.Tensor):
    a1 = torch.reshape(a1, [-1])
    a2 = torch.reshape(a2, [-1])
    return torch.nn.functional.cosine_similarity(a1, a2, dim=0)


left_cos_sim = compare(left, left_original).item()
right_cos_sim = compare(right, right_original).item()
weight_delta_cos_sim = compare(delta_weight, delta_weight_original).item()

print(f"{left_cos_sim=}")
print(f"{right_cos_sim=}")
print(f"{weight_delta_cos_sim=}")