from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline


@dataclass(kw_only=True)
class GeneratePrefixesHparams:
    seperator: str = ". "
    num_tokens: int = 10
    num_sequences: int = 10


def generate_prefixes(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    hparams: GeneratePrefixesHparams | Iterable[GeneratePrefixesHparams],
) -> tuple[np.ndarray, ...]:
    return tuple(iter_generate_prefixes(
        model=model,
        tokenizer=tokenizer,
        hparams=hparams))


def iter_generate_prefixes(*,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    hparams: GeneratePrefixesHparams | Iterable[GeneratePrefixesHparams],
) -> Iterator[np.ndarray]:
    if isinstance(hparams, Iterable):
        for hparams_ in hparams:
            yield from iter_generate_prefixes(
                model=model,
                tokenizer=tokenizer,
                hparams=hparams_)
        return

    pipe = pipeline("text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        truncation=True,
        max_length=hparams.num_tokens,
        num_return_sequences=hparams.num_sequences)
    for result in pipe(""):
        text = result["generated_text"]
        text = text + hparams.seperator
        tokens = tokenizer.encode(text)
        tokens = np.asarray(tokens, dtype=np.int64)
        yield tokens
