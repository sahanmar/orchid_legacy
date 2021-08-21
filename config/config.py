import json
import sys

from dataclasses import dataclass
from typing import Dict, List, cast, Set, Any
from pathlib import Path

from torch.functional import split

from utils.util_types import EncodingType, TensorType, Optional

VALID_TENSOR_TYPES = {"pt": TensorType.torch, "tf": TensorType.tensorFlow, "np": TensorType.numpy}


@dataclass(frozen=True)
class DataPaths:
    train: Path
    test: Path
    dev: Path

    @staticmethod
    def load_config(cfg: Dict[str, str]) -> "DataPaths":
        return DataPaths(**{k: Path(v) for k, v in cfg.items()})


@dataclass(frozen=True)
class ModelCfg:
    dev_mode: bool
    train: bool
    params: Dict[str, int]
    batch_size: int
    split_value: float
    training_folder: Path
    enable_cuda: bool

    @staticmethod
    def load_config(cfg: Dict[str, Any]) -> "ModelCfg":
        return ModelCfg(
            cfg["dev_mode"],
            cfg["train"],
            cfg["params"],
            cfg["batch_size"],
            cfg["split_value"],
            Path(cfg["training_folder"]),
            cfg["enable_cuda"],
        )


@dataclass(frozen=True)
class EncodingCfg:
    encoding_type: EncodingType

    @staticmethod
    def load_config(cfg: Dict[str, str]) -> "EncodingCfg":

        mapper = {"SpanBERT_base_cased": EncodingType.SpanBERT_base_cased}

        return EncodingCfg(encoding_type=mapper[cfg["encoding_type"]])


@dataclass(frozen=True)
class CacheCfg:
    path: Path
    tensor_type: TensorType

    @staticmethod
    def load_config(cfg: Dict[str, str]) -> Optional["CacheCfg"]:
        if not cfg:
            return None
        return CacheCfg(Path(cfg["path"]), VALID_TENSOR_TYPES.get(cfg["tensor_type"], TensorType.torch))


@dataclass(frozen=True)
class TextCfg:
    correference_tags: Set[str]

    @staticmethod
    def load_config(cfg: Dict[str, List[str]]):
        return TextCfg(set(cfg.get("correference_tags", set())))


@dataclass(frozen=True)
class Config:
    data_path: DataPaths
    model: ModelCfg
    encoding: EncodingCfg
    cache: Optional[CacheCfg]
    text: TextCfg

    @staticmethod
    def load_config(config_path: Path) -> "Config":
        if not config_path.is_file():
            print(f"The path '{config_path}' is not a file")
            sys.exit()
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return Config(
            **cast(
                Dict[str, Any],  # type: ignore
                {
                    "data_path": DataPaths.load_config(cfg["data"]),
                    "model": ModelCfg.load_config(cfg["model"]),
                    "encoding": EncodingCfg.load_config(cfg["encoding"]),
                    "cache": CacheCfg.load_config(cfg["cache"]),
                    "text": TextCfg.load_config(cfg["text"]),
                },
            )
        )
