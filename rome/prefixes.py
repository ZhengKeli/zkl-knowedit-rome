from typing import Iterable, Iterator

from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from .utils.generate import generate_fast


def iter_random_prefixes(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    length_params: Iterable[tuple[int, int]],
) -> Iterator[str]:
    yield ""
    for text in iter_random_texts(model, tokenizer, length_params):
        yield text + ". "


def iter_random_texts(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    length_params: Iterable[tuple[int, int]],
) -> Iterator[str]:
    for length, n_gen in length_params:
        yield from generate_fast(model, tokenizer, ["<|endoftext|>"], n_gen_per_prompt=n_gen, max_out_len=length)
