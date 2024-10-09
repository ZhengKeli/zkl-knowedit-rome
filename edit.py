import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from zkl_serialization import load_and_parse_json

from rome import RomeHparams, TextRomeRewriting, TokenizedRomeRewriting, apply_rome_to_model, compute_c_inv, \
    make_default_prefixes, make_default_preservings
from rome.compute_c_v2 import RomeComputeCHParams, compute_c

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

print("Generating pre-update text")
pipe = pipeline("text-generation",
    model=model,
    tokenizer=tokenizer,
    device=model.device,
    num_return_sequences=1,
    return_full_text=True,
    max_new_tokens=64)
pre_update_text = [pipe(prompt)[0]['generated_text'] for prompt in generation_prompts]
del pipe
print(pre_update_text)

print(f"Applying ROME to model")

rewriting_tokenized = TokenizedRomeRewriting.from_text_rewriting(rewriting, tokenizer)
prefixes_tokenized = make_default_prefixes(model, tokenizer)
preservings_tokenized = make_default_preservings(tokenizer, rewriting_tokenized)


def dataset_iterator():
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


c = compute_c(
    RomeComputeCHParams(
        total_tokens_num=100000,
        batch_samples_num=4,
        context_tokens_num=256),
    model,
    model.get_submodule(hparams.rewrite_module_name),
    dataset_iterator())
c_inv = torch.inverse(c)

apply_rome_to_model(model, hparams, rewriting_tokenized, prefixes_tokenized, preservings_tokenized, c_inv)

print("Generating post-update text")
pipe = pipeline("text-generation",
    model=model,
    tokenizer=tokenizer,
    device=model.device,
    num_return_sequences=1,
    return_full_text=True,
    max_new_tokens=64)
post_update_text = [pipe(prompt)[0]['generated_text'] for prompt in generation_prompts]
del pipe
print(post_update_text)

print("Summarizing differences")
for i, (prompt, pre, post) in enumerate(zip(generation_prompts, pre_update_text, post_update_text)):
    if i > 0:
        print("".join(["-" for _ in range(10)]))

    prompt_str = "[Prompt]:"
    pre_str = f"[Pre]:"
    post_str = f"[Post]:"
    pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

    for s, t in zip([prompt_str, pre_str, post_str], [prompt, pre, post]):
        print(s.ljust(pad_to), t)
