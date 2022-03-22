import torch
import numpy as np
import hashlib

from typing import Dict, List, TypeVar, Union, Optional, Tuple
from itertools import groupby, chain
from transformers import AutoTokenizer, AutoModel, AutoConfig  # type: ignore
from tqdm import tqdm
from transformers.models import bert

from config.config import EncodingCfg
from data_processing.cacher import Cacher
from utils.util_types import EncodingType, TensorType, TokenRange
from utils.utils import out_of_menu_exit
from utils.util_types import ConllSentence

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CONTEXT = {
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "dtype": torch.float32,
}

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
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer, config: AutoConfig):
        self.model = model.to(CONTEXT["device"])
        self.tokenizer = tokenizer
        self.config = config

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'(' \
               f'model={self.model.__class__.__name__},' \
               f'tokenizer={self.tokenizer.__class__.__name__}' \
               f')'

    @staticmethod
    def from_config(config: EncodingCfg) -> "GeneralisedBertEncoder":

        if config.encoding_type.value not in ENCODER_MAPPER:
            out_of_menu_exit(text="encoder")

        encoder = ENCODER_MAPPER[config.encoding_type.value]

        model = AutoModel.from_pretrained(encoder)
        tokenizer = AutoTokenizer.from_pretrained(encoder)
        bert_config = AutoConfig.from_pretrained(encoder)

        return GeneralisedBertEncoder(model, tokenizer, bert_config)

    def __call__(
        self, tokens: List[str], tensors_type: TensorType = TensorType.torch
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, List[List[int]]]]:

        """
        This method works with tokenized text. This is done based on OntoNotes input.
        The method returns a dict with encoded ids and tensors.
        """
        with torch.no_grad():
            if tensors_type.value not in TENSOR_MAPPER:
                out_of_menu_exit(text="tensor type")
            bpe_tokens = [self.tokenizer(token)["input_ids"][1:-1] for token in tokens]
            flatten_pbe_tokens = list(chain.from_iterable(bpe_tokens))
            torch_pbe_tokens = torch.unsqueeze(
                torch.tensor([101] + flatten_pbe_tokens + [102]).to(CONTEXT["device"]),  # type: ignore
                dim=0,
            )
            tensors = self.model(
                input_ids=torch_pbe_tokens,
                attention_mask=torch.ones((1, len(torch_pbe_tokens))).to(CONTEXT["device"], dtype=torch.long),  # type: ignore
                token_type_ids=torch.zeros((1, len(torch_pbe_tokens))).to(  # type: ignore
                    CONTEXT["device"], dtype=torch.long
                ),
            )

        original_tokens = []
        counter = 0
        for token in bpe_tokens:
            bpe_tokens = []
            for _ in token:
                bpe_tokens.append(counter)
                counter += 1
            original_tokens.append(bpe_tokens)

        return {
            "input_ids": flatten_pbe_tokens,
            "tensors": tensors["last_hidden_state"][:, 1:-1, :],
            "original_tokens": original_tokens,
        }

    def encode_many(
        self,
        tokenized_sentences: List[List[str]],
        tensors_type: TensorType = TensorType.torch,
        cacher: Optional[Cacher] = None,
    ) -> List[Dict[str, Union[torch.Tensor, np.ndarray, List[List[int]]]]]:
        encoded_sentences: List[Dict[str, Union[torch.Tensor, np.ndarray, List[List[int]]]]] = []
        if cacher is None:
            return [self(sentence, tensors_type) for sentence in tqdm(tokenized_sentences)]
        for i, sentence in tqdm(enumerate(tokenized_sentences)):
            hashed_text = naive_sha1_hash(i, sentence)
            encoded_sent = cacher.get_from_cache(hashed_text)  # type: ignore
            if encoded_sent is None:
                encoded_sent = self(sentence, tensors_type)
                cacher.create_cache(hashed_text, encoded_sent)
            encoded_sentences.append(encoded_sent)
        return encoded_sentences

    def get_cached(self, hash: str) -> Optional[Dict[str, Union[Tensor, List[List[int]]]]]:
        return None


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


def to_doc_based_batches(
    tensors_2_batch: List[torch.Tensor], doc_id_array: List[int], batch_size: int
) -> List[torch.Tensor]:
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
    return [
        torch_custom_padding(doc_tensors[i_start:i_end], lengths[i_start:i_end])
        for i_start, i_end in zip(*get_batch_idxs(len(doc_tensors), batch_size))
    ]


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


def text_based_span_and_corref_tokens_shift(
    text: List[ConllSentence], doc_ids: List[int]
) -> Tuple[List[List[TokenRange]], List[Dict[int, List[TokenRange]]]]:
    # TODO: doc-based global token ids for coref labels
    text_spans = []
    shift_correfs = []
    doc_id_w_sent_id = [(doc_i, sent_i) for sent_i, doc_i in enumerate(doc_ids)]
    sliced_sent_ids = [list(group[1]) for group in groupby(doc_id_w_sent_id, key=lambda x: x[0])]
    doc = [[text[sent_id] for _, sent_id in sent_ids] for sent_ids in sliced_sent_ids]
    for doc_sents in doc:
        sentence_lengths = list(np.cumsum([len(sent.word_tokens) for sent in doc_sents[:-1]]))
        sentence_lengths.insert(0, 0)
        sentence_spans = [list(chain.from_iterable(sent.spans.values())) for sent in doc_sents]
        sentence_correfs = [
            [(corref, label) for label, correfs in sent.coreferences.items() for corref in correfs]
            for sent in doc_sents
        ]
        text_spans.append(
            [span + shift for shift, spans in zip(sentence_lengths, sentence_spans) for span in spans]
        )
        shift_correfs.append(
            [
                (corref + shift, label)
                for shift, correfs in zip(sentence_lengths, sentence_correfs)
                for corref, label in correfs
            ]
        )

    grouped_shift_correfs = [
        {
            label: [tr for tr, _ in values]
            for label, values in groupby(sorted(correfs, key=lambda x: x[1]), key=lambda x: x[1])
        }
        for correfs in shift_correfs
    ]
    return text_spans, grouped_shift_correfs


def get_batch_idxs(array_size: int, batch_size: int) -> Tuple[List[int], List[int]]:
    # The method returns start and end idxs lists to split the array to batches
    if array_size <= batch_size:
        return [0], [array_size]
    slice_idx = [i for i in range(array_size) if i % batch_size == 0]
    slice_idx.append(array_size)
    return slice_idx[:-1], slice_idx[1:]
