import json
import sys
from abc import ABCMeta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, cast, Set, Any, Union

import torch

from utils.env import get_env_variables, resolve_env
from utils.log import get_stream_logger
from utils.types import TensorType, Optional

_logger = get_stream_logger('config')

# Constants
SPEAKER_START = 49518  # 'Ġ#####'
SPEAKER_END = 22560  # 'Ġ###'
NULL_ID_FOR_COREF = 0

# Types
VALID_TENSOR_TYPES = {"pt": TensorType.pt, "tf": TensorType.tf, "np": TensorType.np}

# Environment-based configuration
env = resolve_env()
__shared_env_path = Path(__file__).resolve().parents[1].joinpath(f'.env.{env}')
__secret_env_path = Path(__file__).resolve().parents[1].joinpath(f'.env.secret')
env_config = get_env_variables(
    path_shared=__shared_env_path,
    path_secret=__secret_env_path if __secret_env_path.exists() and __secret_env_path.is_file() else None
)


def _infer_device() -> str:
    output = "cpu"
    use_cuda = env_config.get("USE_CUDA")
    if use_cuda is not None and int(use_cuda) and torch.cuda.is_available():
        output = "cuda"
    return output


class AbstractOrchidConfig(metaclass=ABCMeta):
    class AbsentEnvironmentVariable(Exception):
        def __init__(self, var_name: str):
            super().__init__(
                f'\"{var_name}\" is not specified as the environment variable'
            )

    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        else:
            raise AttributeError(
                f'{self.__class__.__name__} does not have attribute \"{item}\"'
            )

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "AbstractOrchidConfig":
        return AbstractOrchidConfig(**cfg)

    @classmethod
    def _get_absolute_path_from_env(cls, var_name: str) -> Path:
        path = env_config.get(var_name)
        if path is None:
            raise cls.AbsentEnvironmentVariable(var_name)
        path = Path(path).expanduser().resolve()
        return path


@dataclass(frozen=True, init=False)
class Context(AbstractOrchidConfig):
    device = _infer_device()
    dtype = torch.float32
    n_gpu: int = torch.cuda.device_count()


@dataclass(frozen=True)
class DataPaths(AbstractOrchidConfig):
    train: Path
    test: Path
    dev: Path

    @classmethod
    def from_dict(cls, cfg: Dict[str, str]) -> "DataPaths":
        return DataPaths(
            **{
                k: cls.get_absolute_path_to_data(data_file_rel_path=v)
                for k, v in map(lambda kv: (kv[0], Path(kv[1])), cfg.items())
            }
        )

    @classmethod
    def get_absolute_path_to_data(cls, data_file_rel_path: Union[str, Path]) -> Path:
        path_ = cls._get_absolute_path_from_env(var_name='DATA_DIR')
        path_ = path_.joinpath(data_file_rel_path)
        # Make sure that data exist
        if not path_.exists() or not path_.is_file():
            raise FileNotFoundError(f'Invalid input data path: {str(path_)}')
        return path_


@dataclass(frozen=True)
class ModelParameters(AbstractOrchidConfig):
    embeds_dim: int
    trainable_embeddings: bool
    max_span_length: int
    top_lambda: float
    dropout_prob: float
    ffnn_size: int
    normalise_loss: bool
    batch_size: int


@dataclass(frozen=True)
class TrainingConfig(AbstractOrchidConfig):
    split_value: float
    training_epochs: int
    head_learning_rate: float
    weight_decay: float
    learning_rate: float
    warmup_steps: int
    adam_epsilon: float
    adam_beta1: float
    adam_beta2: float
    amp: bool
    fp16_opt_level: str
    local_rank: int
    gradient_accumulation_steps: int
    seed: int
    logging_steps: int
    training_folder: str = str(
        AbstractOrchidConfig._get_absolute_path_from_env(var_name='OUTPUT_DIR')
    )
    evaluation_folder: str = str(
        AbstractOrchidConfig._get_absolute_path_from_env(var_name='OUTPUT_DIR')
    )


@dataclass(frozen=True)
class ModelConfig(AbstractOrchidConfig):
    dev_mode: bool
    train: bool
    eval: bool
    params: ModelParameters
    training: TrainingConfig

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "ModelConfig":
        return ModelConfig(
            dev_mode=cfg["dev_mode"],
            train=cfg["train"],
            eval=cfg["eval"],
            params=ModelParameters(**cfg["params"]),
            training=TrainingConfig(**cfg["training"]),
        )


class EncoderMapping(Enum):
    SpanBERT_base_cased: str = "SpanBERT/spanbert-base-cased"
    SpanBERT_large_cased: str = "SpanBERT/spanbert-large-cased"
    longformer_large_4096: str = "allenai/longformer-large-4096"


@dataclass(frozen=True)
class EncoderConfig(AbstractOrchidConfig):
    encoder_path: EncoderMapping
    max_seq_len: int

    @staticmethod
    def from_dict(cfg: Dict[str, str]) -> "EncoderConfig":
        return EncoderConfig(
            EncoderMapping[cfg["encoding_type"]],
            int(cfg["max_seq_len"])
        )


@dataclass(frozen=True)
class TextConfig(AbstractOrchidConfig):
    coreference_tags: Set[str]
    max_total_seq_len: int

    @staticmethod
    def from_dict(cfg: Dict[str, List[str]]):
        return TextConfig(
            set(cfg.get("coreference_tags", set())),
            cfg.get("max_total_seq_len", -1)
        )


@dataclass(frozen=True)
class Config(AbstractOrchidConfig):
    data_path: DataPaths
    model: ModelConfig
    encoding: EncoderConfig
    text: TextConfig

    @staticmethod
    def _load_from_json(path: Path) -> Dict[str, Any]:
        if path is None or not path.is_file():
            _logger.error(f"\"{str(path)}\" is not a file")
            sys.exit()
        with open(path, "r") as f:
            data = json.load(f)
            return data

    @classmethod
    def from_path(
            cls,
            config_path: Optional[Path] = None,
    ) -> "Config":
        if config_path is None:
            config_path = cls._get_absolute_path_from_env('CONFIG_PATH')
            _logger.info(
                f'Reading configuration from {str(config_path)}'
            )
        cfg = cls._load_from_json(path=config_path)
        return Config(
            **cast(
                Dict[str, Any],  # type: ignore
                {
                    "data_path": DataPaths.from_dict(cfg["data_path"]),
                    "model": ModelConfig.from_dict(cfg["model"]),
                    "encoding": EncoderConfig.from_dict(cfg["encoding"]),
                    "text": TextConfig.from_dict(cfg["text"]),
                },
            )
        )
