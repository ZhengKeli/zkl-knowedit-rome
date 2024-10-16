from dataclasses import dataclass

import numpy as np
from transformers import PreTrainedTokenizer
from typing_extensions import overload


@dataclass
class TextRewriting:
    prompt: str
    target: str
    subject_head: int
    subject_tail: int

    @overload
    def __init__(self, *,
        prompt: str,
        target: str,
        subject: str,
    ):
        pass

    @overload
    def __init__(self, *,
        prompt: str,
        target: str,
        subject_head: int,
        subject_tail: int,
    ):
        pass

    def __init__(self, *,
        prompt: str,
        target: str,
        subject: str | None = None,
        subject_head: int | None = None,
        subject_tail: int | None = None,
    ):
        self.prompt = prompt
        self.target = target

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
class TokenizedRewriting:
    prompt: np.ndarray
    target: np.ndarray
    subject_head: int
    subject_tail: int

    @property
    def subject(self) -> np.ndarray:
        return self.prompt[self.subject_head:self.subject_tail]

    @classmethod
    def from_text_rewriting(cls, rewriting: TextRewriting, tokenizer: PreTrainedTokenizer):
        prefix_tokenized = np.asarray(tokenizer.encode(rewriting.prompt[:rewriting.subject_head]), dtype=np.int64)
        suffix_tokenized = np.asarray(tokenizer.encode(rewriting.prompt[rewriting.subject_tail:]), dtype=np.int64)
        subject_tokenized = np.asarray(tokenizer.encode(rewriting.subject), dtype=np.int64)
        target_tokenized = np.asarray(tokenizer.encode(rewriting.target), dtype=np.int64)
        prompt_tokenized = np.concatenate([prefix_tokenized, subject_tokenized, suffix_tokenized])

        subject_head_tokenized = len(prefix_tokenized)
        subject_tail_tokenized = subject_head_tokenized + len(subject_tokenized)

        return cls(
            prompt=prompt_tokenized,
            target=target_tokenized,
            subject_head=subject_head_tokenized,
            subject_tail=subject_tail_tokenized)
