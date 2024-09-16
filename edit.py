import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from zkl_serialization import load_and_parse_json

from rome import ROMEHyperParams, apply_rome_to_model
from util.generate import generate_fast

model_name = "gpt2-medium"
hparams_file_path = os.path.join("hparams/ROME/gpt2-medium.json")

requests = [{
    'prompt': "{} was the founder of",
    'subject': "Steve Jobs",
    'target_new': {"str": "Microsoft"},
}]

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
hparams = load_and_parse_json(hparams_file_path, ROMEHyperParams)
print(hparams)

print("Generating pre-update text")
pre_update_text = generate_fast(model, tokenizer, generation_prompts, max_out_len=100)
print(pre_update_text)

print(f"Applying ROME to model")
model, _ = apply_rome_to_model(model, tokenizer, requests, hparams)

print("Generating post-update text")
post_update_text = generate_fast(model, tokenizer, generation_prompts, max_out_len=100)
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
