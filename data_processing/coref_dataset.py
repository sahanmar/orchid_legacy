import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Tuple,
    List,
    Optional,
    Iterable,
    NewType,
    Union,
    Any
)

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

from config import (
    SPEAKER_START,
    SPEAKER_END,
    VERBOSITY,
    NULL_ID_FOR_COREF,
    Config
)
from utils.general import flatten_list_of_lists
from utils.log import get_stream_logger

CorefCluster = NewType('CorefCluster', List[Union[List[int], Tuple[int, int]]])
RawExample = NewType('RawExample', Tuple[Any, List[str], List[CorefCluster], List[str]])
ModelInputExample = NewType('Example', Tuple[torch.Tensor, ...])
ModelInputMetadata = NewType('ModelInputMetadata', Tuple[str, List[int]])

logger = get_stream_logger(Path(__file__).stem, verbosity=VERBOSITY)


@dataclass
class PreEncodedExample:
    doc_key: str
    end_token_idx_to_word_idx: List[int]
    token_ids: List[int]
    clusters: CorefCluster


# noinspection DuplicatedCode
class CorefDataset(Dataset):

    def __init__(
            self,
            data_path: Path,
            tokenizer_path: str,
            null_coref_id: int = NULL_ID_FOR_COREF
    ):
        self._tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        self.model_max_length = self.tokenizer.init_kwargs.get(
            'model_max_length',
            512
        )
        self.null_coref_id = null_coref_id

        logger.info(f'Parsing input data from \"{str(data_path)}\"')
        assert data_path.exists() and \
               data_path.is_file() and \
               data_path.suffix.endswith('jsonl'), f'Incorrect input data file'
        (
            self.examples,
            self.max_mention_num,
            self.max_cluster_size,
            self.max_num_clusters
        ) = self._parse_jsonl(path=data_path)
        (
            self.examples,
            self.lengths,
            self.num_examples_filtered
        ) = self._pre_encode(self.examples)
        logger.info(
            f'The dataset has been created: '
            f'size={len(self.examples)}, '
            f'filtered_out={self.num_examples_filtered}'
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'(' \
               f'examples={len(self.examples)},' \
               f'max_mention_num={self.max_mention_num},' \
               f'max_cluster_size={self.max_cluster_size},' \
               f'max_num_clusters={self.max_num_clusters},' \
               f'num_examples_filtered={self.num_examples_filtered},' \
               f'max_seq_length={self.max_mention_num}' \
               f'tokenizer={self.tokenizer.__class__.__name__}' \
               f')'

    @staticmethod
    def _parse_jsonl(path: Path) -> Tuple[List[RawExample], int, int, int]:
        examples = []
        max_mention_num = -1
        max_cluster_size = -1
        max_num_clusters = -1
        with open(path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                doc_key = d["doc_key"]
                input_words = flatten_list_of_lists(d["sentences"])
                clusters = d["clusters"]
                max_mention_num = max(max_mention_num, len(flatten_list_of_lists(clusters)))
                max_cluster_size = max(max_cluster_size, max(len(cluster) for cluster in clusters) if clusters else 0)
                max_num_clusters = max(max_num_clusters, len(clusters) if clusters else 0)
                speakers = flatten_list_of_lists(d["speakers"])
                examples.append(
                    RawExample((doc_key, input_words, clusters, speakers))
                )
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _pre_encode(
            self,
            examples: List[RawExample]
    ) -> Tuple[List[PreEncodedExample], List[int], int]:
        coref_examples = []
        lengths = []
        num_examples_filtered = 0
        # Process words and speakers
        # doc_key - document_id
        # words - list of words
        # clusters - coreference clusters
        #  (list of clusters, each cluster is a list of word
        #  spans for specific entity mentions)
        # speakers - list of speaker_ids per word
        for doc_key, words, clusters, speakers in examples:
            e = PreEncodedExample(
                doc_key=doc_key,
                end_token_idx_to_word_idx=[0],  # for <s>
                token_ids=[],
                clusters=clusters
            )
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()

            last_speaker = None
            for idx, (word, speaker) in enumerate(zip(words, speakers)):
                if last_speaker != speaker:
                    # Speaker encoding
                    speaker_prefix_tokenized: List[int] = \
                        [SPEAKER_START] + \
                        self.tokenizer.encode(
                            " " + speaker,
                            add_special_tokens=False
                        ) + \
                        [SPEAKER_END]
                    last_speaker = speaker
                else:
                    speaker_prefix_tokenized: List[int] = []
                for _ in range(len(speaker_prefix_tokenized)):
                    e.end_token_idx_to_word_idx.append(idx)
                e.token_ids.extend(speaker_prefix_tokenized)
                word_idx_to_start_token_idx[idx] = len(e.token_ids) + 1  # +1 for <s>
                word_tokenized = self.tokenizer.encode(
                    " " + word,
                    add_special_tokens=False
                )
                for _ in range(len(word_tokenized)):
                    e.end_token_idx_to_word_idx.append(idx)
                e.token_ids.extend(word_tokenized)
                # old_seq_len + 1 (for <s>)
                # + len(tokenized_word)
                # - 1 (we start counting from zero) = len(token_ids)
                word_idx_to_end_token_idx[idx] = len(e.token_ids)

            if 0 < self.model_max_length < len(e.token_ids):
                num_examples_filtered += 1
                continue

            # Coreference cluster spans within token_ids
            try:
                e.clusters = [
                    [
                        (
                            word_idx_to_start_token_idx[max(0, start - 1)],
                            word_idx_to_end_token_idx[end]
                        )
                        for start, end in cluster
                    ]
                    for cluster in e.clusters
                ]
                for i, cluster in enumerate(e.clusters):
                    logger.debug(f'======= {doc_key} - cluster_{i} =======')
                    for start, end in cluster:
                        logger.debug(
                            f'\"{self.tokenizer.decode(e.token_ids[start:end])}\"'
                        )
            except KeyError:
                raise
            lengths.append(len(e.token_ids))

            coref_examples.append(e)
        return coref_examples, lengths, num_examples_filtered

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item: int) -> PreEncodedExample:
        return self.examples[item]

    def pad_clusters_inside(
            self,
            clusters: List[CorefCluster]
    ) -> List[CorefCluster]:
        return [
            CorefCluster(
                cluster + [(self.null_coref_id, self.null_coref_id)] *
                (self.max_cluster_size - len(cluster))
            ) for cluster in clusters
        ]

    def pad_clusters_outside(
            self, clusters: List[CorefCluster]
    ) -> List[CorefCluster]:
        return clusters + \
               [CorefCluster([])] * (self.max_num_clusters - len(clusters))

    def pad_clusters(self, clusters: List[CorefCluster]) -> List[CorefCluster]:
        clusters = self.pad_clusters_outside(clusters)
        clusters = self.pad_clusters_inside(clusters)
        return clusters

    def pad_batch(
            self,
            batch: List[PreEncodedExample],
            max_length: Optional[int] = None
    ) -> Tuple[List[ModelInputMetadata], ModelInputExample]:
        if max_length is None:
            max_length = self.model_max_length
        # Two additional special tokens <s>, </s>
        padded_batch = []
        input_metadata = []
        for example in batch:
            encoded_dict = self.tokenizer.prepare_for_model(
                example.token_ids,
                add_special_tokens=True,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                prepend_batch_axis=True,
                return_tensors='pt'
            )
            clusters = self.pad_clusters(
                [
                    # Move starts and ends by plus one due to the <s> token
                    CorefCluster([(s + 1, e + 1) for s, e in cluster])
                    for cluster in example.clusters
                ]
            )
            padded_batch.append((
                encoded_dict["input_ids"],
                encoded_dict["attention_mask"],
                torch.tensor(clusters)
            ))
            input_metadata.append(
                ModelInputMetadata(
                    (example.doc_key, example.end_token_idx_to_word_idx)
                )
            )
        tensor_batch = ModelInputExample(tuple(
            torch.stack(
                [example[i].squeeze() for example in padded_batch],
                dim=0
            )
            for i in range(len(padded_batch[-1] if padded_batch else 0))
        ))
        return input_metadata, tensor_batch


# noinspection DuplicatedCode
class BucketBatchSampler(DataLoader):

    def __init__(
            self,
            dataset: CorefDataset,
            batch_size_1: bool = False,
    ):
        super(BucketBatchSampler, self).__init__(dataset=dataset)
        self.data_source = dataset
        self.data_source.examples.sort(key=lambda x: len(x.token_ids),
                                       reverse=True)
        self.max_seq_len = min(self.data_source.model_max_length,
                               len(dataset[0].token_ids))
        self.batches = self.prepare_batches() if not batch_size_1 else self.prepare_eval_batches()

    def prepare_batches(self) -> List[Tuple[List[ModelInputMetadata], ModelInputExample]]:
        batches = []
        batch = []
        per_example_batch_len = 0
        for example in self.data_source:
            if len(batch) == 0:
                # TODO change to config.attention_window
                per_example_batch_len = self.calc_effective_per_example_batch_len(len(example.token_ids))
            elif (len(batch) + 1) * per_example_batch_len > self.max_seq_len:
                batch = self.data_source.pad_batch(batch, max_length=self.max_seq_len)
                batches.append(batch)
                batch = []
                per_example_batch_len = self.calc_effective_per_example_batch_len(len(example.token_ids))
            batch.append(example)
        if len(batch) == 0:
            return batches
        batch = self.data_source.pad_batch(batch, max_length=self.max_seq_len)
        batches.append(batch)
        return batches

    def __iter__(self) -> Iterable[Tuple[Tuple[str, List[int]], ModelInputExample]]:
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    @staticmethod
    def calc_effective_per_example_batch_len(example_len: int) -> int:
        return math.ceil((example_len + 2) / 512) * 512

    def prepare_eval_batches(
            self
    ) -> List[Tuple[List[ModelInputMetadata], ModelInputExample]]:
        batches = []
        for example in self.data_source:
            max_length = min(len(example.token_ids), self.max_seq_len)
            batch = self.data_source.pad_batch([example], max_length=max_length)
            batches.append(batch)
        return batches


def get_dataset(data_type: str, config: Config) -> CorefDataset:
    # TODO: add caching
    try:
        path = config.data_path[data_type]
    except AttributeError:
        logger.error('Could not get the correct data path')
        raise
    coref_dataset = CorefDataset(
        data_path=path,
        tokenizer_path=config.encoding.encoder_path.value
    )

    return coref_dataset
