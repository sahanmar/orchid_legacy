import json
import math
import random
from collections import namedtuple
from pathlib import Path
from typing import (
    Tuple,
    List,
    Optional,
    Iterable

)

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

from config import (
    SPEAKER_START,
    SPEAKER_END,
    NULL_ID_FOR_COREF,
    Config
)
from utils.general import flatten_list_of_lists
from utils.log import get_stream_logger

CorefExample = namedtuple("CorefExample", ["token_ids", "clusters"])

logger = get_stream_logger(Path(__file__).stem)


# noinspection DuplicatedCode
class CorefDataset(Dataset):

    def __init__(
            self,
            data_path: Path,
            tokenizer_path: str,
            max_seq_length: Optional[int] = -1
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_seq_length = max_seq_length

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
        ) = self._tokenize(self.examples)
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
    def _parse_jsonl(path: Path) -> Tuple[List, int, int, int]:
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
                examples.append((doc_key, input_words, clusters, speakers))
        return examples, max_mention_num, max_cluster_size, max_num_clusters

    def _tokenize(self, examples):
        coref_examples = []
        lengths = []
        num_examples_filtered = 0
        for doc_key, words, clusters, speakers in examples:
            word_idx_to_start_token_idx = dict()
            word_idx_to_end_token_idx = dict()
            end_token_idx_to_word_idx = [0]  # for <s>

            token_ids = []
            last_speaker = None
            for idx, (word, speaker) in enumerate(zip(words, speakers)):
                if last_speaker != speaker:
                    speaker_prefix = \
                        [SPEAKER_START] + \
                        self.tokenizer.encode(
                            " " + speaker,
                            add_special_tokens=False
                        ) + \
                        [SPEAKER_END]
                    last_speaker = speaker
                else:
                    speaker_prefix = []
                for _ in range(len(speaker_prefix)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(speaker_prefix)
                word_idx_to_start_token_idx[idx] = len(token_ids) + 1  # +1 for <s>
                tokenized = self.tokenizer.encode(" " + word, add_special_tokens=False)
                for _ in range(len(tokenized)):
                    end_token_idx_to_word_idx.append(idx)
                token_ids.extend(tokenized)
                # old_seq_len + 1 (for <s>)
                # + len(tokenized_word)
                # - 1 (we start counting from zero) = len(token_ids)
                word_idx_to_end_token_idx[idx] = len(token_ids)

            if 0 < self.max_seq_length < len(token_ids):
                num_examples_filtered += 1
                continue

            new_clusters = [
                [(word_idx_to_start_token_idx[start], word_idx_to_end_token_idx[end]) for start, end in cluster] for
                cluster in clusters]
            lengths.append(len(token_ids))

            coref_examples.append(
                ((doc_key, end_token_idx_to_word_idx), CorefExample(token_ids=token_ids, clusters=new_clusters)))
        return coref_examples, lengths, num_examples_filtered

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    def pad_clusters_inside(self, clusters):
        return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (self.max_cluster_size - len(cluster)) for cluster
                in clusters]

    def pad_clusters_outside(self, clusters):
        return clusters + [[]] * (self.max_num_clusters - len(clusters))

    def pad_clusters(self, clusters):
        clusters = self.pad_clusters_outside(clusters)
        clusters = self.pad_clusters_inside(clusters)
        return clusters

    def pad_batch(self, batch, max_length: int):
        max_length += 2  # we have additional two special tokens <s>, </s>
        padded_batch = []
        for example in batch:
            encoded_dict = self.tokenizer.encode_plus(
                # TODO: do this in a better way; this is just a fix
                #  for the non-functional code from the parent s2e repo
                self.tokenizer.decode(
                    example[0],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                ),
                add_special_tokens=True,
                padding='max_length',
                max_length=max_length,
                return_attention_mask=True,
                return_tensors='pt'
            )
            clusters = self.pad_clusters(example.clusters)
            example = (encoded_dict["input_ids"], encoded_dict["attention_mask"]) + (torch.tensor(clusters),)
            padded_batch.append(example)
        tensored_batch = tuple(
            torch.stack([example[i].squeeze() for example in padded_batch], dim=0) for i in range(len(example)))
        return tensored_batch


# noinspection DuplicatedCode
class BucketBatchSampler(DataLoader):

    def __init__(
            self,
            dataset: CorefDataset,
            max_seq_len: int,
            max_total_seq_len: int,
            batch_size_1: bool = False,
    ):
        super(BucketBatchSampler, self).__init__(dataset=dataset)
        self.data_source = dataset
        dataset.examples.sort(key=lambda x: len(x[1].token_ids), reverse=True)
        self.max_seq_len = min(max_seq_len, len(dataset[0][1].token_ids))
        self.max_total_seq_len = max_total_seq_len
        self.batches = self.prepare_batches() if not batch_size_1 else self.prepare_eval_batches()

    def prepare_batches(self):
        batches = []
        batch = []
        per_example_batch_len = 0
        for _, elem in self.data_source:
            if len(batch) == 0:
                # TODO change to config.attention_window
                per_example_batch_len = self.calc_effective_per_example_batch_len(len(elem.token_ids))
            elif (len(batch) + 1) * per_example_batch_len > self.max_total_seq_len:
                batch = self.data_source.pad_batch(batch, max_length=self.max_seq_len)
                batches.append(batch)
                batch = []
                per_example_batch_len = self.calc_effective_per_example_batch_len(len(elem.token_ids))
            batch.append(elem)
        if len(batch) == 0:
            return batches
        batch = self.data_source.pad_batch(batch, max_length=self.max_seq_len)
        batches.append(batch)
        return batches

    def __iter__(self) -> Iterable[List[int]]:
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def calc_effective_per_example_batch_len(self, example_len):
        return math.ceil((example_len + 2) / 512) * 512

    def prepare_eval_batches(self):
        batches = []
        for doc_key, elem in self.data_source:
            max_length = min(len(elem.token_ids), self.max_seq_len)
            # print(max_length)
            batch = self.data_source.pad_batch(
                [elem],
                max_length=max_length
            )
            batches.append((doc_key, batch))
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
        tokenizer_path=config.encoding.encoder_path.value,
        max_seq_length=config.text.max_total_seq_len
    )

    return coref_dataset
