import sys
import torch
import numpy as np
import hashlib

from typing import Dict, List, TypeVar, Union, Optional
from itertools import groupby, chain
from transformers import AutoTokenizer, AutoModel  # type: ignore

from config.config import EncodingCfg
from data_processing.cacher import Cacher
from utils.util_types import EncodingType, TensorType
from utils.utils import out_of_menu_exit
from utils.util_types import ConllSentence

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

INCONSISTENCY_TOKENS = {"'"}

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
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, List[List[int]]]]:

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
            "original_tokens": self._bpe_to_original_tokens_indices(pbe_tokens),
        }

    def encode_many(
        self,
        tokenized_sentences: List[List[str]],
        tensors_type: TensorType = TensorType.torch,
        cacher: Optional[Cacher] = None,
    ) -> List[Dict[str, Union[torch.Tensor, np.ndarray, List[List[int]]]]]:
        encoded_sentences: List[Dict[str, Union[torch.Tensor, np.ndarray, List[List[int]]]]] = []
        if cacher is None:
            return [self(sentence, tensors_type) for sentence in tokenized_sentences]
        for i, sentence in enumerate(tokenized_sentences):
            hashed_text = naive_sha1_hash(i, sentence)
            encoded_sent = cacher.get_from_cache(hashed_text)  # type: ignore
            if encoded_sent is None:
                encoded_sent = self(sentence, tensors_type)
                cacher.create_cache(hashed_text, encoded_sent)
            encoded_sentences.append(encoded_sent)
        return encoded_sentences

    def get_cached(self, hash: str) -> Optional[Dict[str, Union[Tensor, List[List[int]]]]]:
        return None

    @staticmethod
    def _bpe_to_original_tokens_indices(pbe_tokens: List[str]) -> List[List[int]]:
        """
        Please keep in mind that this method works for BERT based models only
        """
        if not pbe_tokens:
            return []
        tokens: List[List[int]] = [[0]]
        token_to_extend = 0
        for i, pbe_token in enumerate(pbe_tokens[1:], start=1):
            # TODO FIX INCONSISTENCIES
            # Pay attention to inconsistency tokens. There are still inconsistencies
            # but they are minore and do not influence the final result.
            # E.g. original_tokens = [city, 's] -> bpe_to_original = [city', s]
            if pbe_token.startswith("##") or pbe_token in INCONSISTENCY_TOKENS:
                tokens[token_to_extend].append(i)
                continue
            token_to_extend += 1
            tokens.append([i])

        return tokens


def bpe_to_original_embeddings(
    encoded_tokens: Union[torch.Tensor, np.ndarray], bpe_indices: List[List[int]]
) -> Union[torch.Tensor, np.ndarray]:

    # TODO ADD TENSORFLOW IMPLEMENTATION
    # TODO ADD NUMPY IMPLEMENTATION

    if isinstance(encoded_tokens, torch.Tensor):
        batch, _, embed_size = encoded_tokens.size()
        original_tokens_embeddings = torch.zeros((batch, len(bpe_indices), embed_size))
        for i, idices in enumerate(bpe_indices):
            original_tokens_embeddings[0, i, :] = torch.mean(
                torch.stack([encoded_tokens[0, j, :] for j in idices], dim=1), dim=1
            )
        return original_tokens_embeddings

    raise TypeError("Smth is wrong with types...")


def bpe_to_original_embeddings_many(
    encoded_tokens_per_sentences: List[Dict[str, Union[torch.Tensor, np.ndarray, List[List[int]]]]]
) -> List[torch.Tensor]:

    # TODO ADD TENSORFLOW IMPLEMENTATION
    # TODO ADD NUMPY IMPLEMENTATION

    encoded_tokens_list = [
        sent["tensors"]
        for sent in encoded_tokens_per_sentences
        if isinstance(sent["tensors"], (torch.Tensor, np.ndarray))
    ]
    list_bpe_indices = [
        sent["original_tokens"]
        for sent in encoded_tokens_per_sentences
        if not isinstance(sent["original_tokens"], (torch.Tensor, np.ndarray))
    ]

    if isinstance(encoded_tokens_list[0], torch.Tensor):
        orig_token_embeds = []
        for encoded_tokens, bpe_indices in zip(encoded_tokens_list, list_bpe_indices):
            tensor = bpe_to_original_embeddings(encoded_tokens, bpe_indices)
            if isinstance(tensor, torch.Tensor):
                orig_token_embeds.append(tensor)

        assert len(orig_token_embeds) == len(list_bpe_indices)

        return orig_token_embeds

    raise TypeError("Smth is wrong with types...")


def to_doc_based_batches(tensors_2_batch: List[torch.Tensor], doc_id_array: List[int]) -> torch.Tensor:
    doc_id_w_sent_id = [(doc_i, sent_i) for sent_i, doc_i in enumerate(doc_id_array)]
    sliced_sent_ids = [list(group[1]) for group in groupby(doc_id_w_sent_id, key=lambda x: x[0])]

    doc_tensors = [
        torch_custom_flatten(
            [tensors_2_batch[sent_id] for _, sent_id in sent_ids],
            [tensors_2_batch[sent_id].size()[1] for _, sent_id in sent_ids],
        )
        for sent_ids in sliced_sent_ids
    ]
    lengths = [t.size()[1] for t in doc_tensors]
    return torch_custom_padding(doc_tensors, lengths)


def torch_custom_flatten(tensors_2_flatten: List[torch.Tensor], lengths: List[int]) -> torch.Tensor:
    _, _, embed_size = tensors_2_flatten[0].size()
    flatten_tensor = torch.zeros(1, sum(lengths), embed_size)
    shift = 0
    for length, tensor in zip(lengths, tensors_2_flatten):
        flatten_tensor[0, shift : shift + length, :] = tensor[0, :, :]
        shift += length

    return flatten_tensor


def torch_custom_padding(tensors_2_pad: List[torch.Tensor], lengths: List[int]) -> torch.Tensor:
    # I have killed some time with looking for some accurate solution.
    # Maybe I'm a dumbass, however I didn't find anything suitable. Thus,
    # decided to write it on my own. If smb knows a better solution let me
    # know
    max_len_2_pad = max(lengths)
    texts_num = len(tensors_2_pad)
    _, _, embed_dim = tensors_2_pad[0].size()
    padded_tensor = torch.zeros((texts_num, max_len_2_pad, embed_dim))
    for i, (tensor, t_len) in enumerate(zip(tensors_2_pad, lengths)):
        padded_tensor[i, :t_len, :] = tensor
    return padded_tensor


def naive_sha1_hash(index: int, text: List[str]) -> str:
    return hashlib.sha1("".join([str(index), *text]).encode()).hexdigest()


def text_based_span_tokens_shift(text: List[ConllSentence], doc_ids: List[int]) -> List[List[List[int]]]:
    text_spans = []
    doc_id_w_sent_id = [(doc_i, sent_i) for sent_i, doc_i in enumerate(doc_ids)]
    sliced_sent_ids = [list(group[1]) for group in groupby(doc_id_w_sent_id, key=lambda x: x[0])]
    doc = [[text[sent_id] for _, sent_id in sent_ids] for sent_ids in sliced_sent_ids]
    for doc_sents in doc:
        sentence_lengths = [len(sent.word_tokens) for sent in doc_sents[:-1]]
        sentence_lengths.insert(0, 0)
        sentence_spans = [list(chain.from_iterable(sent.spans.values())) for sent in doc_sents]
        text_spans.append(
            [
                [val + shift for val in span]
                for shift, spans in zip(sentence_lengths, sentence_spans)
                for span in spans
            ]
        )
    return text_spans
