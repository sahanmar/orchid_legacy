import torch
import numpy as np
import csv

from pathlib import Path
from typing import Dict, List, TypeVar, Union, Optional

from config.config import CacheCfg
from utils.util_types import TensorType

Tensor = TypeVar("Tensor", torch.Tensor, np.ndarray)


class Cacher:
    def __init__(self, cache_path: Path, tensor_type: TensorType):
        mkdir_if_not_exist(cache_path)

        self.path = cache_path
        self.tensor_type = tensor_type
        self.cached_ids = set(el.stem() for el in self.path.iterdir() if el.is_dir())

    @staticmethod
    def from_config(config: CacheCfg) -> "Cacher":
        return Cacher(config.path, config.tensor_type)

    def create_cache(self, hash: str, encoded_instance: Dict[str, Union[Tensor, List[List[int]]]]) -> None:
        instance_dir_path = self.path / hash
        mkdir_if_not_exist(instance_dir_path)
        tensor_dir_path = instance_dir_path / self.tensor_type.value
        mkdir_if_not_exist(tensor_dir_path)
        self.write_tensors(tensor_dir_path / "input_ids", encoded_instance["input_ids"])
        self.write_tensors(tensor_dir_path / "tensors", encoded_instance["tensors"])
        self.write_original_tokens_ids(
            instance_dir_path / "original_tokens.csv", encoded_instance["original_tokens"]
        )

    def get_from_cache(self, hash: str) -> Optional[Dict[str, Union[Tensor, List[List[int]]]]]:
        if hash not in self.cached_ids:
            return None
        instance_dir_path = self.path / hash
        tensor_dir_path = instance_dir_path / self.tensor_type.value
        return {
            "input_ids": self.load_tensor(tensor_dir_path / "input_ids"),
            "tensors": self.load_tensor(tensor_dir_path / "tensors"),
            "original_tokens": self.load_original_tokens_ids(instance_dir_path / "original_tokens.csv"),
        }

    def write_tensors(self, path: Path, tensor: Tensor) -> None:
        if self.tensor_type == TensorType.torch:
            torch.save(tensor, path.parent / (path.name + ".pt"))
        else:
            raise Exception("Not implemented for this type of tensor...")

    def load_tensor(self, path) -> Union[Tensor, List[List[int]]]:
        if self.tensor_type == TensorType.torch:
            return torch.load(path.parent / (path.name + ".pt"))
        else:
            raise Exception("Not implemented for this type of tensor...")

    @staticmethod
    def write_original_tokens_ids(path: Path, token_ids: List[List[int]]) -> None:
        with open(path, "w") as f:
            wr = csv.writer(f)
            wr.writerows(token_ids)

    @staticmethod
    def load_original_tokens_ids(path: Path) -> List[List[int]]:
        with open(path, "r") as f:
            rows = csv.reader(f)
            data = list(zip(*rows))
        return [[int(bpe_id) for bpe_id in token] for token in data]


def mkdir_if_not_exist(path: Path) -> None:
    if not path.is_dir():
        path.mkdir()
