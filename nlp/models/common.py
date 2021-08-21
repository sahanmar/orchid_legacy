import torch

from typing import List, Tuple


def batch_split_idx(array_size: int, split_value: float) -> int:
    return int(array_size * split_value)
