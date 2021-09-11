import torch

from re import I
from typing import List, Optional, Union, Dict

from nlp.encoder import (
    GeneralisedBertEncoder,
    bpe_to_original_embeddings_many,
    to_doc_based_batches,
    text_based_span_and_corref_tokens_shift,
    get_batch_idxs,
)
from data_processing.conll_parses import ConllParser
from data_processing.cacher import Cacher
from config.config import Config, ModelCfg

from nlp.models.torch.e2ecr import E2ECR, Trainer, create_target_values
from nlp.models import batch_split_idx

from utils.util_types import PipelineOutput, Response

context = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "dtype": torch.float32}


class OrchidPipeline:
    def __init__(
        self,
        data_loader: ConllParser,
        encoder: GeneralisedBertEncoder,
        corref_config: ModelCfg,
        cacher: Optional[Cacher],
    ):
        self.data_loader = data_loader
        self.encoder = encoder
        self.corref_config = corref_config
        self.cacher = cacher

    @staticmethod
    def from_config(config: Config) -> "OrchidPipeline":
        return OrchidPipeline(
            ConllParser.from_config(config),
            GeneralisedBertEncoder.from_config(config.encoding),
            config.model,
            Cacher.from_config(config.cache) if config.cache is not None else None,
        )

    def __call__(self):
        # try:
        if 1:
            # Load Data
            sentences = self.data_loader()

            # Encode
            sentences_texts = [[token.text for token in sent.word_tokens] for sent in sentences]
            doc_ids = [sent.document_index for sent in sentences]
            text_spans, grouped_shift_correfs = text_based_span_and_corref_tokens_shift(sentences, doc_ids)

            encoded_tokens_per_sentences = self.encoder.encode_many(sentences_texts, cacher=self.cacher)
            orig_encoded_tokens_per_sentences = bpe_to_original_embeddings_many(encoded_tokens_per_sentences)

            # Batch the data
            doc_based_batches = to_doc_based_batches(
                orig_encoded_tokens_per_sentences, doc_ids, self.corref_config.batch_size
            )
            text_spans_batches = [
                text_spans[i_start:i_end]
                for i_start, i_end in zip(*get_batch_idxs(len(text_spans), self.corref_config.batch_size))
            ]

            # Model Initializing, Training, Inferencing
            model = E2ECR(**self.corref_config.params).to(context["device"])
            print(model)
            if self.corref_config.train:
                # TODO ADD TESTS!
                target_values_batches = [
                    create_target_values(text_spans[i_start:i_end], grouped_shift_correfs[i_start:i_end])
                    for i_start, i_end in zip(*get_batch_idxs(len(text_spans), self.corref_config.batch_size))
                ]
                split_inx = batch_split_idx(len(doc_based_batches), self.corref_config.split_value)
                train_docs, train_span_ids, train_target = (
                    doc_based_batches[:split_inx],
                    text_spans_batches[:split_inx],
                    target_values_batches[:split_inx],
                )
                test_docs, test_span_ids, test_target = (
                    doc_based_batches[split_inx:],
                    text_spans_batches[split_inx:],
                    target_values_batches[split_inx:],
                )
                # Initialize Trainer
                trainer = Trainer(model)

                # trainer.train(
                #     train_data=(train_docs, train_span_ids, train_target),
                #     test_data=(test_docs, test_span_ids, test_target),
                #     folder_to_save=self.corref_config.training_folder,
                #     num_epochs=5,
                # )

            model_out = [
                model(doc_based_batch, text_spans_batch)
                for doc_based_batch, text_spans_batch in zip(doc_based_batches, text_spans_batches)
            ]

            return PipelineOutput(state=Response.success)

        # except:  # must specify the error type
        #    return PipelineOutput(state=Response.fail)
