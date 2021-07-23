from typing import Optional

from nlp.encoder import GeneralisedBertEncoder
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
            encoded_sentences = self.encoder.encode_many(sentences_texts, cacher=self.cacher)

            # Model Initializing, Training, Inferencing
            model = E2ECR(**self.coref_config.params)

            return PipelineOutput(state=Response.success)

        except:  # must specify the error type
            return PipelineOutput(state=Response.fail)
