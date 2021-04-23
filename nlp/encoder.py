import sys
import torch
import numpy as np

from typing import Dict, List, TypeVar, Union, Optional
from transformers import AutoTokenizer, AutoModel  # type: ignore

from config.config import EncodingCfg

from utils.util_types import EncodingType, TensorType
from utils.utils import out_of_menu_exit


ENCODER_MAPPER = {
    EncodingType.SpanBERT_base_cased.value: "SpanBERT/spanbert-base-cased",
    EncodingType.SpanBERT_large_cased.value: "SpanBERT/spanbert-large-cased",
}

TENSOR_MAPPER = {
    TensorType.torch.value: "pt",
    TensorType.tensorFlow.value: "tf",
    TensorType.numpy.value: "np",
}

Tensor = TypeVar("Tensor", torch.Tensor, np.ndarray)


class GeneralisedBertEncoder:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def from_config(config: EncodingCfg) -> "GeneralisedBertEncoder":

        if config.encoding_type.value not in ENCODER_MAPPER:
            out_of_menu_exit(text="encoder")

        encoder = ENCODER_MAPPER[config.encoding_type.value]

        model = AutoModel.from_pretrained(encoder)
        tokenizer = AutoTokenizer.from_pretrained(encoder)

        return GeneralisedBertEncoder(model, tokenizer)

    def __call__(
        self, tokens: List[str], tensors_type: TensorType = TensorType.torch
    ) -> Dict[str, Union[Tensor, List[List[int]]]]:

        """
        This method works with tokenized text. This is done based on OntoNotes input. 
        The method returns a dict with encoded ids and tensors. 
        """

        if tensors_type.value not in TENSOR_MAPPER:
            out_of_menu_exit(text="tensor type")

        pbe_tokens = self.tokenizer.tokenize(tokens, is_split_into_words=True)
        tokenized = self.tokenizer(
            tokens, return_tensors=TENSOR_MAPPER[tensors_type.value], is_split_into_words=True
        )
        tensors = self.model(**tokenized)

        return {
            "input_ids": tokenized["input_ids"],
            "tensors": tensors["last_hidden_state"],
            "original_tokens": self._pbe_to_original_tokens_indices(pbe_tokens),
        }

    def encode_many(
        self, tokenized_sentences: List[List[str]], tensors_type: TensorType = TensorType.torch
    ) -> List[Dict[str, Union[Tensor, List[List[int]]]]]:
        encoded_sentences: List[Dict[str, Union[Tensor, List[List[int]]]]] = []
        for i, sentence in enumerate(tokenized_sentences):
            encoded_sent = self.get_cached(naive_hash(i, sentence))
            encoded_sentences.append(
                encoded_sent if encoded_sent else self(sentence, tensors_type)  # type: ignore
            )
        return encoded_sentences

    def get_cached(self, hash: str) -> Optional[Dict[str, Union[Tensor, List[List[int]]]]]:
        return None

    @staticmethod
    def _pbe_to_original_tokens_indices(pbe_tokens: List[str]) -> List[List[int]]:
        """
        Please keep in mind that this method works for BERT based models only
        """
        if not pbe_tokens:
            return []
        tokens: List[List[int]] = [[0]]
        token_to_extend = 0
        for i, pbe_token in enumerate(pbe_tokens[1:], start=1):
            if pbe_token.startswith("##"):
                tokens[token_to_extend].append(i)
                continue
            token_to_extend += 1
            tokens.append([i])

        return tokens


def naive_hash(index: int, text: List[str]) -> str:
    return "".join([str(index), *text])
