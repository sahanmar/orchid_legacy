from pathlib import Path
from typing import Tuple, Optional, List, Dict
from utils.util_types import ConllSentence, TokenRange, Morphology, CorrefTokenType

from config.config import Config


class ConllParser:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self) -> List[ConllSentence]:
        texts = self.split_file_2_texts(self.path)

        parsed_sentences: List[ConllSentence] = []
        sent_i = 0
        for text in texts:
            for sent_i, sentence in enumerate(text):
                parsed_sentences.append(self.process_sentence(sentence, sent_i))

        return parsed_sentences

    @staticmethod
    def from_config(config: Config) -> "ConllParser":
        path = (
            config.data_path.dev
            if config.model.dev_mode
            else config.data_path.train
            if config.model.train
            else config.data_path.test
        )
        return ConllParser(path)

    def add_corref_mutable(
        self,
        corref_dict: Dict[int, List[TokenRange]],
        unprocessed_corref: str,
        i_token: int,
    ) -> None:
        for corref in unprocessed_corref.split("|"):
            cr_label, cr_type = self.parse_cr(corref)
            if cr_label not in corref_dict:
                corref_dict[cr_label] = []
            if cr_type == CorrefTokenType.full:
                corref_dict[cr_label].append(TokenRange(start=i_token, end=i_token + 1))
            elif cr_type == CorrefTokenType.start:
                corref_dict[cr_label].append(TokenRange(start=i_token, end=i_token))
            else:
                corref_dict[cr_label][-1].end = i_token + 1

    def process_sentence(self, sentence: List[str], sent_index) -> ConllSentence:

        tokens: List[Morphology] = []
        correferences: Dict[int, List[TokenRange]] = {}

        for per_token_annotations in sentence:
            (
                folder,
                i_sent_str,
                i_token_str,
                token,
                pos,
                _,
                lemma,
                _,
                _,
                speaker,
                *rest,
            ) = per_token_annotations.split()
            i_token = int(i_token_str)

            cr = rest[-1]
            tokens.append(Morphology(text=token, part_of_speech=pos, lemma=lemma if lemma != "-" else token))

            if not cr.startswith("-"):
                self.add_corref_mutable(correferences, cr, i_token)

        return ConllSentence(
            folder=folder,
            sentence_index=int(sent_index),
            word_tokens=tokens,
            speaker=speaker,
            correferences=correferences,
        )

    @staticmethod
    def split_file_2_texts(path: Path) -> List[List[List[str]]]:

        texts: List[List[List[str]]] = []
        sentence: List[str] = []
        text: List[List[str]] = []

        with open(path, "r") as f:
            for line in f.readlines():
                if line.startswith("#begin document"):
                    continue
                if line.startswith("#end document"):
                    texts.append(text)
                    text = []
                    continue
                if not line.strip():
                    text.append(sentence)
                    del sentence
                    sentence = []
                    continue
                sentence.append(line)
        return texts

    @staticmethod
    def parse_cr(text: str) -> Tuple[int, CorrefTokenType]:
        if text.startswith("("):
            if text.endswith(")"):
                return int(text[1:-1]), CorrefTokenType.full
            return int(text[1:]), CorrefTokenType.start
        return int(text[:-1]), CorrefTokenType.end
