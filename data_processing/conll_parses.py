import re

from pathlib import Path
from typing import Tuple, List, Dict
from itertools import groupby

from utils.util_types import ConllSentence, TokenRange, Morphology, CorrefTokenType
from config.config import Config


span_types_to_parse = re.compile(r"\(")
span_end = re.compile(r"\)")
span_key_pat = re.compile(r"[a-zA-Z]+")


class ConllParser:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self) -> List[ConllSentence]:
        texts = self.split_file_2_texts(self.path)

        parsed_sentences: List[ConllSentence] = []
        sent_i = 0
        for sentence in texts:
            if sentence == "art_break":
                sent_i = 0
            parsed_sentences.append(self.process_sentence(sentence, sent_i))
            sent_i += 1

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
        spans: List[Tuple[List[int], str]] = []

        num_of_spans = 0
        tmp_spans: List[Tuple[List[int], str]] = []

        for per_token_annotations in sentence:
            (
                folder,
                _,
                i_token_str,
                token,
                pos,
                span,
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

            # TODO
            # Make it beautiful

            num_of_new_spans = len(span_types_to_parse.findall(span))
            num_of_spans += num_of_new_spans
            if num_of_spans > 0:
                # span_keys = span.split("(")  # [1 : num_of_new_spans + 1]
                tmp_spans += [
                    ([], token.strip("*"))
                    for _, token in zip(range(num_of_spans), span_key_pat.findall(span))
                ]
                for span_ids in tmp_spans:
                    span_ids[0].append(i_token)
            num_of_spans_to_dump = len(span_end.findall(span))
            num_of_spans -= num_of_spans_to_dump
            if num_of_spans_to_dump > 0:
                for span_ids in tmp_spans[-num_of_spans_to_dump:]:
                    spans.append(span_ids)
                tmp_spans = tmp_spans[:-num_of_spans_to_dump]

        sorted_spans = sorted(spans, key=lambda x: x[1])
        grouped_spans = {
            key: [s_idxs for s_idxs, _ in span_idxs_w_keys]
            for key, span_idxs_w_keys in groupby(sorted_spans, key=lambda x: x[1])
        }

        return ConllSentence(
            folder=folder,
            sentence_index=int(sent_index),
            word_tokens=tokens,
            speaker=speaker,
            correferences=correferences,
            spans=grouped_spans,
        )

    @staticmethod
    def split_file_2_texts(path: Path) -> List[List[str]]:

        texts: List[List[str]] = []
        sentence: List[str] = []

        with open(path, "r") as f:
            for line in f.readlines():
                if line.startswith("#begin document"):
                    continue
                if line.startswith("#end document"):
                    sentence.append("art_break")
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
