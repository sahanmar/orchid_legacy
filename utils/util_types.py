from dataclasses import dataclass
from typing import Optional, Any, Callable


@dataclass
class ConllDataType:
    text: str
    word_tokens: str
    speaker: str
    coreferences:


@dataclass
class Corefence:
    label: int,
    coref_range: TokenRange

