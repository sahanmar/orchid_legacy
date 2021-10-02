from dataclasses import dataclass
from enum import Enum

from typing import Optional, Callable, Dict, List, Union


class EncodingType(Enum):
    SpanBERT_base_cased = 1
    SpanBERT_large_cased = 2


class TensorType(Enum):
    torch = 1
    tensorFlow = 2
    numpy = 3


class CorrefTokenType(Enum):
    """
    Correferences are labeled as spans. Thus, more than one
    consecutive words can be assigned to the same correference.
    We distinguish
    start <- the begging of the span
    end <- the end of the span
    full <- the correference is one token
    """

    start = 1
    end = 2
    full = 3


class Response(Enum):
    success = 0
    fail = 1


class TokenRange:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end  # including this token
        assert self.start < self.end

    def __repr__(self):
        return f"({self.start},{self.end})"

    def __add__(self, shift: int) -> "TokenRange":
        return TokenRange(self.start + shift, self.end + shift)

    def to_consecutive_list(self):
        return list(range(self.start, self.end + 1))

    def inside(self, token_range: "TokenRange") -> bool:
        return self.start <= token_range.start and self.end >= token_range.end

    @staticmethod
    def from_list(list_of_consecutive_elements: List[int]) -> "TokenRange":
        return TokenRange(list_of_consecutive_elements[0], list_of_consecutive_elements[-1] + 1)


@dataclass
class Morphology:
    text: str
    part_of_speech: str
    lemma: str


@dataclass
class ConllSentence:
    folder: str
    document_index: int
    word_tokens: List[Morphology]
    speaker: str
    correferences: Dict[int, List[TokenRange]]
    spans: Dict[str, List[TokenRange]]


@dataclass
class PipelineOutput:
    state: Response
