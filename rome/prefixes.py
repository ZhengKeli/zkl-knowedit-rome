from typing import Iterable, Iterator

from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline


def iter_random_prefixes(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    length_params: Iterable[tuple[int, int]],
) -> Iterator[str]:
    yield ""
    for text in iter_random_texts(model, tokenizer, length_params):
        yield text + ". "


def iter_random_texts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    length_params: Iterable[tuple[int, int]],
) -> Iterator[str]:
    for length, n_gen in length_params:
        pipe = pipeline("text-generation",
            model=model,
            tokenizer=tokenizer,
            device=model.device,
            num_return_sequences=n_gen,
            return_full_text=False,
            max_new_tokens=length)
        for result in pipe(""):
            yield result["generated_text"]
