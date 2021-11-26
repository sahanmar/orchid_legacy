import torch
import numpy as np

from pathlib import Path
from typing import Dict, List, TypeVar, Union, Optional

from torch import Tensor

from config.config import CacheCfg
from utils.util_types import TensorType

EncodedInstance = TypeVar("EncodedInstance")

CONTEXT = {
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "dtype": torch.float32,
}


class Cacher:
    def __init__(self, cache_path: Path, tensor_type: TensorType):
        mkdir_if_not_exist(cache_path)

        self.path = cache_path
        self.tensor_type = tensor_type
        self.cached_ids = set(el.name for el in self.path.iterdir() if el.is_dir())

    @staticmethod
    def from_config(config: CacheCfg) -> "Cacher":
        return Cacher(config.path, config.tensor_type)

    def create_cache(self, hash: str, encoded_instance: Dict[str, EncodedInstance]) -> None:
        instance_dir_path = self.path / hash
        mkdir_if_not_exist(instance_dir_path)
        tensor_dir_path = instance_dir_path / str(self.tensor_type.value)
        mkdir_if_not_exist(tensor_dir_path)
        self.write_tensors(tensor_dir_path / "input_ids", encoded_instance["input_ids"])
        # self.write_tensors(tensor_dir_path / "tensors", encoded_instance["tensors"])
        self.write_original_tokens_ids(
            instance_dir_path / "original_tokens.txt", encoded_instance["original_tokens"]  # type: ignore
        )
        self.cached_ids = set(el.name for el in self.path.iterdir() if el.is_dir())

    def get_from_cache(self, hash: str) -> Optional[Dict[str, EncodedInstance]]:
        if hash not in self.cached_ids:
            return None
        instance_dir_path = self.path / hash
        tensor_dir_path = instance_dir_path / str(self.tensor_type.value)
        input_ids = self.load_tensor(tensor_dir_path / "input_ids")  # type: ignore
        # tensors = self.load_tensor(tensor_dir_path / "tensors")  # type: ignore
        return {
            "input_ids": input_ids.to(
                CONTEXT["device"] if isinstance(input_ids, torch.Tensor) else input_ids
            ),
            # "tensors": tensors.to(CONTEXT["device"] if isinstance(tensors, torch.Tensor) else tensors),
            "original_tokens": self.load_original_tokens_ids(instance_dir_path / "original_tokens.txt"),  # type: ignore
        }

    def write_tensors(self, path: Path, tensor: EncodedInstance) -> None:
        if self.tensor_type == TensorType.torch:
            torch.save(tensor, path.parent / (path.name + ".pt"))
        else:
            raise Exception("Not implemented for this type of tensor...")

    def load_tensor(self, path) -> EncodedInstance:
        if self.tensor_type == TensorType.torch:
            return torch.load(path.parent / (path.name + ".pt"))
        else:
            raise Exception("Not implemented for this type of tensor...")

    @staticmethod
    def write_original_tokens_ids(path: Path, token_ids: List[List[int]]) -> None:
        with open(path, "w") as f:
            for token in token_ids:
                f.write("%s\n" % " ".join([str(t) for t in token]))

    @staticmethod
    def load_original_tokens_ids(path: Path) -> List[List[int]]:
        token_ids = []
        with open(path, "r") as f:
            for line in f.readlines():
                token_ids.append([int(val) for val in line.strip().split()])
        return token_ids


def mkdir_if_not_exist(path: Path) -> None:
    if not path.is_dir():
        path.mkdir()
