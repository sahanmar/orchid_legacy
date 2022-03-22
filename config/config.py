import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, cast, Set, Any

from utils.env import get_env_configuration, resolve_env
from utils.log import get_stream_logger
from utils.util_types import EncodingType, TensorType, Optional

_logger = get_stream_logger('config')

# Environment-based configuration
env = resolve_env()
__shared_env_path = Path(__file__).resolve().parents[1].joinpath(f'.env.{env}')
__secret_env_path = Path(__file__).resolve().parents[1].joinpath(f'.env.secret')
env_config = get_env_configuration(
    path_shared=__shared_env_path,
    path_secret=__secret_env_path if __secret_env_path.exists() and __secret_env_path.is_file() else None
)

VALID_TENSOR_TYPES = {"pt": TensorType.torch, "tf": TensorType.tensorFlow, "np": TensorType.numpy}


@dataclass(frozen=True)
class DataPaths:
    train: Path
    test: Path
    dev: Path

    @staticmethod
    def from_dict(cfg: Dict[str, str]) -> "DataPaths":
        return DataPaths(
            **{
                k: v if v.is_absolute() else Path(__file__).resolve().parents[1].joinpath(v)
                for k, v in map(lambda kv: (kv[0], Path(kv[1])), cfg.items())
            }
        )


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
    def from_dict(cfg: Dict[str, Any]) -> "ModelCfg":
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
    def from_dict(cfg: Dict[str, str]) -> "EncodingCfg":
        mapper = {"SpanBERT_base_cased": EncodingType.SpanBERT_base_cased}

        return EncodingCfg(encoding_type=mapper[cfg["encoding_type"]])


@dataclass(frozen=True)
class CacheCfg:
    path: Path
    tensor_type: TensorType

    @staticmethod
    def from_dict(cfg: Dict[str, str]) -> Optional["CacheCfg"]:
        if not cfg:
            return None
        return CacheCfg(Path(cfg["path"]), VALID_TENSOR_TYPES.get(cfg["tensor_type"], TensorType.torch))


@dataclass(frozen=True)
class TextCfg:
    coreference_tags: Set[str]

    @staticmethod
    def from_dict(cfg: Dict[str, List[str]]):
        return TextCfg(set(cfg.get("coreference_tags", set())))


@dataclass(frozen=True)
class Config:
    data_path: DataPaths
    model: ModelCfg
    encoding: EncodingCfg
    cache: Optional[CacheCfg]
    text: TextCfg

    @staticmethod
    def _load_from_json(path: Path) -> Dict[str, Any]:
        if path is None or not path.is_file():
            _logger.error(f"\"{path}\" is not a file")
            sys.exit()
        with open(path, "r") as f:
            data = json.load(f)
            return data

    @classmethod
    def from_path(cls, config_path: Path) -> "Config":
        cfg = cls._load_from_json(path=config_path)
        if env != 'test':
            _logger.info(f'Reading secret_configuration')
            cfg_secret = cls._load_from_json(
                path=cast(Path, env_config.get('SECRET_CONFIG_PATH'))
            )
            cfg = cfg.update(cfg_secret)
        else:
            _logger.debug('Ignoring secret configuration')

        # Attempt reading data paths from environment variables
        return Config(
            **cast(
                Dict[str, Any],  # type: ignore
                {
                    "data_path": DataPaths.from_dict(cfg["data"]),
                    "model": ModelCfg.from_dict(cfg["model"]),
                    "encoding": EncodingCfg.from_dict(cfg["encoding"]),
                    "cache": CacheCfg.from_dict(cfg["cache"]),
                    "text": TextCfg.from_dict(cfg["text"]),
                },
            )
        )
