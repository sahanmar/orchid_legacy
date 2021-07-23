from typing import List, Optional, Union, Dict

from nlp.encoder import GeneralisedBertEncoder, Tensor, bpe_to_original_embeddings_many
from data_processing.conll_parses import ConllParser
from data_processing.cacher import Cacher
from config.config import Config, ModelCfg

from nlp.models.torch.e2ecr import E2ECR

from utils.util_types import ConllSentence, PipelineOutput, Response

from itertools import chain


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
            text_spans = text_based_span_tokens_shift(sentences)

            encoded_tokens_per_sentences = self.encoder.encode_many(sentences_texts, cacher=self.cacher)
            orig_encoded_tokens_per_sentences = bpe_to_original_embeddings_many(encoded_tokens_per_sentences)

            # Model Initializing, Training, Inferencing
            model = E2ECR(**self.coref_config.params)

            return PipelineOutput(state=Response.success)

        except:  # must specify the error type
            return PipelineOutput(state=Response.fail)


def text_based_span_tokens_shift(text: List[ConllSentence]) -> List[List[int]]:
    sentence_lengths = [len(sent.word_tokens) for sent in text[:-1]]
    sentence_lengths.insert(0, 0)
    sentence_spans = [list(chain.from_iterable(sent.spans.values())) for sent in text]
    return [
        [val + shift for val in span]
        for shift, spans in zip(sentence_lengths, sentence_spans)
        for span in spans
    ]
