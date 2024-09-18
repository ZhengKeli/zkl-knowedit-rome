from typing import Iterator

from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from .utils.generate import generate_fast

CONTEXT_TEMPLATES_CACHE = None


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        generated_prefixes = iter_random_texts(model, tok, length_params)

        CONTEXT_TEMPLATES_CACHE = ["{}"] + [x + ". {}" for x in generated_prefixes]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def iter_random_texts(
    model: AutoModelForCausalLM,
    tok: PreTrainedTokenizer,
    length_params: tuple[int, int],
) -> Iterator[str]:
    for length, n_gen in length_params:
        yield from generate_fast(model, tok, ["<|endoftext|>"], n_gen_per_prompt=n_gen, max_out_len=length)
