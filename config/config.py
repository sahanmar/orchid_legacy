import json
import sys

from dataclasses import dataclass
from typing import Dict, cast, TypeVar, Any
from pathlib import Path

from utils.util_types import EncodingType


@dataclass(frozen=True)
class DataPaths:
    train: Path
    test: Path
    dev: Path

    @staticmethod
    def load_cfg(cfg: Dict[str, str]):
        return DataPaths(**{k: Path(v) for k, v in cfg.items()})


@dataclass(frozen=True)
class ModelCfg:
    dev_mode: bool
    train: bool


@dataclass(frozen=True)
class EncodingCfg:
    encoding_type: EncodingType

    @staticmethod
    def load_cfg(cfg: Dict[str, str]):

        mapper = {"SpanBERT_base_cased": EncodingType.SpanBERT_base_cased}

        return EncodingCfg(encoding_type=mapper[cfg["encoding_type"]])


@dataclass(frozen=True)
class Config:
    data_path: DataPaths
    model: ModelCfg
    encoding: EncodingCfg

    @staticmethod
    def load_cfg(config_path: Path):
        if not config_path.is_file():
            print(f"The path '{config_path}' is not a file")
            sys.exit()
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return Config(
            **cast(
                Dict[str, Any],  # type: ignore
                {
                    "data_path": DataPaths.load_cfg(cfg["data"]),
                    "model": ModelCfg(**cfg["model"]),
                    "encoding": EncodingCfg.load_cfg(cfg["encoding"]),
                },
            )
        )
