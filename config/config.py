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
class ParametersCfg:
    embeds_dim: int
    max_span_length: int
    top_lambda: float
    dropout_prob: float
    ffnn_size: int
    normalise_loss: bool
    batch_size: int


@dataclass(frozen=True)
class TrainingCgf:
    split_value: float
    training_folder: Path
    training_epochs: int
    head_learning_rate: float
    weight_decay: float
    learning_rate: float
    warmup_steps: int
    adam_beta1: float
    adam_beta2: float
    amp: bool
    fp16_opt_level: str
    local_rank: int
    gradient_accumulation_steps: int
    seed: int
    logging_steps: int
    do_eval: True

    @staticmethod
    def load_config(cfg: Dict[str, Any]) -> "TrainingCgf":
        return TrainingCgf(
            split_value=cfg["split_value"],
            training_folder=Path(cfg["training_folder"]),
            training_epochs=cfg["training_epochs"],
            head_learning_rate=cfg["head_learning_rate"],
            weight_decay=cfg["weight_decay"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg["warmup_steps"],
            adam_beta1=cfg["adam_beta1"],
            adam_beta2=cfg["adam_beta2"],
            amp=cfg["amp"],
            fp16_opt_level=cfg["fp16_opt_level"],
            local_rank=cfg["local_rank"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            seed=cfg["seed"],
            logging_steps=cfg["logging_steps"],
            do_eval=cfg["do_eval"],
        )


@dataclass(frozen=True)
class ModelCfg:
    dev_mode: bool
    train: bool
    params: ParametersCfg
    training: TrainingCgf
    enable_cuda: bool

    @staticmethod
    def load_config(cfg: Dict[str, Any]) -> "ModelCfg":
        return ModelCfg(
            cfg["dev_mode"],
            cfg["train"],
            ParametersCfg(**cfg["params"]),
            TrainingCgf.load_config(cfg["training"]),
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
