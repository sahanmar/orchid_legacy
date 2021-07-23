import json
import sys

from dataclasses import dataclass
from typing import Dict, cast, TypeVar, Any
from pathlib import Path

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
class Config:
    data_path: DataPaths
    model: ModelCfg
    encoding: EncodingCfg
    cache: Optional[CacheCfg]

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
                    "model": ModelCfg(**cfg["model"]),
                    "encoding": EncodingCfg.load_config(cfg["encoding"]),
                    "cache": CacheCfg.load_config(cfg["cache"]),
                },
            )
        )
