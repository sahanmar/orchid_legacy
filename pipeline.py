from re import I
from typing import List, Optional, Union, Dict

from nlp.encoder import (
    GeneralisedBertEncoder,
    bpe_to_original_embeddings_many,
    to_doc_based_batches,
    text_based_span_tokens_shift,
)
from data_processing.conll_parses import ConllParser
from data_processing.cacher import Cacher
from config.config import Config, ModelCfg

from nlp.models.torch.e2ecr import E2ECR

from utils.util_types import PipelineOutput, Response


class OrchidPipeline:
    def __init__(
        self,
        data_loader: ConllParser,
        encoder: GeneralisedBertEncoder,
        coref_config: ModelCfg,
        cacher: Optional[Cacher],
    ):
        self.data_loader = data_loader
        self.encoder = encoder
        self.coref_config = coref_config
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

        try:
            # Load Data
            sentences = self.data_loader()

            # Encode
            sentences_texts = [[token.text for token in sent.word_tokens] for sent in sentences]
            doc_ids = [sent.document_index for sent in sentences]
            text_spans = text_based_span_tokens_shift(sentences, doc_ids)

            encoded_tokens_per_sentences = self.encoder.encode_many(sentences_texts, cacher=self.cacher)
            orig_encoded_tokens_per_sentences = bpe_to_original_embeddings_many(encoded_tokens_per_sentences)

            doc_based_batches = to_doc_based_batches(orig_encoded_tokens_per_sentences, doc_ids)

            # Model Initializing, Training, Inferencing
            model = E2ECR(**self.coref_config.params)
            model_out = model(doc_based_batches, text_spans)

            return PipelineOutput(state=Response.success)

        except:  # must specify the error type
            return PipelineOutput(state=Response.fail)
