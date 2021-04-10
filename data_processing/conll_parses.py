import argparse
import sys

from pathlib import Path
from typing import Tuple, Optional, List, Dict
from utils.util_types import ConllSentence, TokenRange, Morphology, CorrefTokenType

from config.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file", default="config/config.json")
    return parser.parse_args()


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


def parse_cr(text: str) -> Tuple[int, CorrefTokenType]:
    if text.startswith("("):
        if text.endswith(")"):
            return int(text[1:-1]), CorrefTokenType.full
        return int(text[1:]), CorrefTokenType.start
    return int(text[:-1]), CorrefTokenType.end


def add_corref_mutable(
    corref_dict: Dict[int, List[TokenRange]], unprocessed_corref: str, i_token: int
) -> None:
    for corref in unprocessed_corref.split("|"):
        cr_label, cr_type = parse_cr(corref)
        if cr_label not in corref_dict:
            corref_dict[cr_label] = []
        if cr_type == CorrefTokenType.full:
            corref_dict[cr_label].append(TokenRange(start=i_token, end=i_token + 1))
        elif cr_type == CorrefTokenType.start:
            corref_dict[cr_label].append(TokenRange(start=i_token, end=i_token))
        else:
            corref_dict[cr_label][-1].end = i_token + 1


def process_sentences(sentence: List[str]) -> ConllSentence:

    tokens: Dict[int, Morphology] = {}
    correferences: Dict[int, List[TokenRange]] = {}

    for i, per_token_annotations in enumerate(sentence):
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
        tokens[i_token] = Morphology(text=token, part_of_speech=pos, lemma=lemma if lemma != "-" else token)

        if not cr.startswith("-"):
            add_corref_mutable(correferences, cr, i_token)

    return ConllSentence(
        folder=folder,
        sentence_index=int(i_sent_str),
        word_tokens=tokens,
        speaker=speaker,
        correferences=correferences,
    )


def main():
    args = parse_args()
    config = Config.load_cfg(Path(args.config))
    path = (
        config.data_path.dev
        if config.model.dev_mode
        else config.data_path.train
        if config.model.train
        else config.config_path.test
    )

    texts = split_file_2_texts(path)
    processed_sentences = [process_sentences(sentence) for sentence in texts]


if __name__ == "__main__":
    main()
