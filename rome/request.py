from dataclasses import dataclass

from typing_extensions import overload


@dataclass
class TextRomeRequest:
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
