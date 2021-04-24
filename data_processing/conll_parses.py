from pathlib import Path
from typing import Tuple, Optional, List, Dict
from utils.util_types import ConllSentence, TokenRange, Morphology, CorrefTokenType

from config.config import Config


class ConllParser:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self) -> List[ConllSentence]:
        texts = self.split_file_2_texts(self.path)
        return [self.process_sentences(sentence) for sentence in texts]

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
        self, corref_dict: Dict[int, List[TokenRange]], unprocessed_corref: str, i_token: int,
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

    def process_sentences(self, sentence: List[str]) -> ConllSentence:

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
            sentence_index=int(i_sent_str),
            word_tokens=tokens,
            speaker=speaker,
            correferences=correferences,
        )

    @staticmethod
    def split_file_2_texts(path: Path) -> List[List[str]]:

        texts: List[List[str]] = []
        sentence: List[str] = []

        with open(path, "r") as f:
            for line in f.readlines():
                if line.startswith("#begin document") or line.startswith("#end document"):
                    continue
                if not line.strip():
                    texts.append(sentence)
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
