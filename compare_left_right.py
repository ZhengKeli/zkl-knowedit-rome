import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from zkl_rome import RomeComputeCHParams, RomeComputeVDeltaHparams, TextRomeRewriting, TokenizedRomeRewriting, compute_c, \
    compute_left_right, make_default_prefixes, make_default_preservings

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
model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

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


module = model.get_submodule("transformer.h.8.mlp.c_proj")

c = compute_c(
    RomeComputeCHParams(
        total_tokens_num=1000000,
        batch_samples_num=4,
        context_tokens_num=256),
    model, module,
    dataset_iterator())
c_inv = torch.inverse(c)

(left, right) = compute_left_right(
    RomeComputeVDeltaHparams(
        learning_rate=5e-1,
        stopping_steps_num=20,
        stopping_loss_threshold=5e-2,
        rewriting_loss_k=1.0,
        preserving_loss_k=0.0625,
        regularization_loss_k=0.5,
        regularization_constraint_factor=3.0),
    model, module,
    rewriting_tokenized,
    prefixes_tokenized,
    preservings_tokenized,
    c_inv)
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
