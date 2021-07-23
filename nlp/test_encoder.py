import unittest

import torch

from nlp.encoder import GeneralisedBertEncoder
from config.config import EncodingCfg
from utils.util_types import EncodingType


class EncodingTest(unittest.TestCase):
    def test_encdoing(self):
        cfg = EncodingCfg(EncodingType.SpanBERT_base_cased)
        encoding_model = GeneralisedBertEncoder.from_config(cfg)

        text = "It is Saturday and instead of being out I am wring these test cases ."
        tokenized_text = text.split()

        encoded = encoding_model(tokenized_text)

        self.assertEqual(type(encoded["input_ids"]), torch.Tensor)

        # test if ids are correct
        self.assertEqual(
            encoded["input_ids"].squeeze().tolist(),  # type: ignore
            [
                101,
                1122,
                1110,
                2068,
                2149,
                6194,
                1105,
                1939,
                1104,
                1217,
                1149,
                178,
                1821,
                192,
                3384,
                1292,
                2774,
                2740,
                119,
                102,
            ],
        )

        # Test pbe 2 original tokens
        self.assertEqual(len(encoded["original_tokens"]), len(tokenized_text))
