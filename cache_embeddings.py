from pathlib import Path

from config.config import Config, DataPaths
from pipeline import OrchidPipeline
from utils.types import Response
from data_processing.cacher import Cacher
from data_processing.conll_parses import ConllParser
from nlp.encoder import GeneralisedBERTEncoder


def cache_embeds(data_loader, encoder, cacher):
    sentences = data_loader()
    sentences_texts = [[token.text for token in sent.word_tokens] for sent in sentences]
    encoded_tokens_per_sentences = encoder.encode_many(sentences_texts, cacher=cacher)


def main():
	config = Config.from_path(Path("config/config.json"))

	data_loader = ConllParser.from_config(config)
	encoder = GeneralisedBERTEncoder.from_config(config.encoding)
	if config.cache is None:
		raise ValueError("Bro... No chacher defined!")
	cacher = Cacher.from_config(config.cache)

	cache_embeds(data_loader, encoder, cacher)


if __name__=="__main__":
	main()
