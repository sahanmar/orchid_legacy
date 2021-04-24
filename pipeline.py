from typing import Optional

from nlp.encoder import GeneralisedBertEncoder
from data_processing.conll_parses import ConllParser
from data_processing.cacher import Cacher
from config.config import Config

from utils.util_types import PipelineOutput, Response


class OrchidPipeline:
    def __init__(self, data_loader: ConllParser, encoder: GeneralisedBertEncoder, cacher: Optional[Cacher]):
        self.data_loader = data_loader
        self.encoder = encoder
        self.cacher = cacher

    @staticmethod
    def from_config(config: Config) -> "OrchidPipeline":
        return OrchidPipeline(
            ConllParser.from_config(config),
            GeneralisedBertEncoder.from_config(config.encoding),
            Cacher.from_config(config.cache) if config.cache is not None else None,
        )

    def __call__(self):

        try:
            # Load Data
            sentences = self.data_loader()

            # Encode
            sentences_texts = [[token.text for token in sent.word_tokens] for sent in sentences]
            encoded_sentences = self.encoder.encode_many(sentences_texts, cacher=self.cacher)

            return PipelineOutput(state=Response.success)

        except:  # must specify the error type
            return PipelineOutput(state=Response.fail)
