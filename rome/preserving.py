from dataclasses import dataclass
from typing import overload

import numpy as np
from transformers import PreTrainedTokenizer

from .rewriting import TokenizedRomeRewriting


@dataclass
class TextRomePreserving:
    prompt: str
    subject_head: int
    subject_tail: int

    @overload
    def __init__(self, *,
        prompt: str,
        subject: str,
    ):
        pass

    @overload
    def __init__(self, *,
        prompt: str,
        subject_head: int,
        subject_tail: int,
    ):
        pass

    def __init__(self, *,
        prompt: str,
        subject: str | None = None,
        subject_head: int | None = None,
        subject_tail: int | None = None,
    ):
        self.prompt = prompt

        if subject_head is not None and subject_tail is not None:
            self.subject_head = subject_head
            self.subject_tail = subject_tail
        elif subject is not None:
            self.subject_head = prompt.rfind(subject)
            if self.subject_head == -1:
                raise ValueError("subject not found")
            self.subject_tail = self.subject_head + len(subject)

    @property
    def subject(self) -> str:
        return self.prompt[self.subject_head:self.subject_tail]

    @property
    def prompt_template(self) -> str:
        return self.prompt[:self.subject_head] + "{}" + self.prompt[self.subject_tail:]


@dataclass
class TokenizedRomePreserving:
    prompt: np.ndarray
    subject_head: int
    subject_tail: int

    @property
    def subject(self) -> np.ndarray:
        return self.prompt[self.subject_head:self.subject_tail]

    @classmethod
    def from_text_preserving(cls, preserving: TextRomePreserving, tokenizer: PreTrainedTokenizer):
        prefix_tokenized = np.asarray(tokenizer.encode(preserving.prompt[:preserving.subject_head]), dtype=np.int64)
        suffix_tokenized = np.asarray(tokenizer.encode(preserving.prompt[preserving.subject_tail:]), dtype=np.int64)
        subject_tokenized = np.asarray(tokenizer.encode(preserving.subject), dtype=np.int64)
        prompt_tokenized = np.concatenate([prefix_tokenized, subject_tokenized, suffix_tokenized])

        subject_head_tokenized = len(prefix_tokenized)
        subject_tail_tokenized = subject_head_tokenized + len(subject_tokenized)

        return cls(
            prompt=prompt_tokenized,
            subject_head=subject_head_tokenized,
            subject_tail=subject_tail_tokenized)


def make_default_preservings(
    tokenizer: PreTrainedTokenizer,
    rewriting: TokenizedRomeRewriting,
) -> tuple[TokenizedRomePreserving, ...]:
    preserving = TokenizedRomePreserving(
        prompt=np.concatenate([
            rewriting.subject,
            tokenizer.encode(" is a")]),
        subject_head=0,
        subject_tail=len(rewriting.subject))
    return preserving,
