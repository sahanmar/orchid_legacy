from dataclasses import dataclass
from enum import Enum

from typing import Optional, Callable, Dict, List


class EncodingType(Enum):

    SpanBERT = 1
    RoBERTa = 2


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


class TokenRange:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"({self.start},{self.end})"


@dataclass
class Morphology:
    text: str
    part_of_speech: str
    lemma: str


@dataclass
class ConllSentence:
    folder: str
    sentence_index: int
    word_tokens: Dict[int, Morphology]
    speaker: str
    correferences: Dict[int, TokenRange]
