import re

from pathlib import Path
from typing import Optional, Set, Tuple, List, Dict
from itertools import groupby

from utils.util_types import ConllSentence, TokenRange, Morphology, CorrefTokenType
from config.config import Config


span_types_to_parse = re.compile(r"\(")
span_end = re.compile(r"\)")
span_key_pat = re.compile(r"[a-zA-Z]+")


class ConllParser:
    def __init__(self, path: Path, coref_to_leave: Optional[Set[str]]):
        self.path = path
        self.coref_to_leave = coref_to_leave if coref_to_leave else set()

    def __call__(self) -> List[ConllSentence]:
        texts = self.split_file_2_texts(self.path)

        parsed_sentences: List[ConllSentence] = []
        doc_i = 0
        for doc_i, text in enumerate(texts):
            for sentence in text:
                parsed_sentences.append(self.process_sentence(sentence, doc_i))

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
        coref_to_leave = config.text.correference_tags
        return ConllParser(path, coref_to_leave)

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

    def process_sentence(self, sentence: List[str], document_index: int) -> ConllSentence:

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

            # Extract all spans

            num_of_new_spans = len(span_types_to_parse.findall(span))
            num_of_spans += num_of_new_spans
            if num_of_spans > 0:
                tmp_spans += [
                    ([], token) for _, token in zip(range(num_of_spans), span_key_pat.findall(span))
                ]
                for span_ids in tmp_spans:
                    span_ids[0].append(i_token)
            num_of_spans_to_dump = len(span_end.findall(span))
            num_of_spans -= num_of_spans_to_dump
            if num_of_spans_to_dump > 0:
                for span_ids in tmp_spans[-num_of_spans_to_dump:]:
                    spans.append(span_ids)
                tmp_spans = tmp_spans[:-num_of_spans_to_dump]

        # Group spans with respect to the span types
        sorted_spans = sorted(spans, key=lambda x: x[1])
        grouped_spans = {
            key: [TokenRange.from_list(s_idxs) for s_idxs, _ in span_idxs_w_keys]
            for key, span_idxs_w_keys in groupby(sorted_spans, key=lambda x: x[1])
            if key in self.coref_to_leave or not self.coref_to_leave
        }

        return ConllSentence(
            folder=folder,
            document_index=document_index,
            word_tokens=tokens,
            speaker=speaker,
            correferences=correferences,
            spans=grouped_spans,
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
