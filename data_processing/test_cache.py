import unittest
import torch

from pathlib import Path

from data_processing.cacher import Cacher
from utils.types import TensorType


class TestCahcer(unittest.TestCase):
    def test_cacher(self):
        path = Path("cache")
        hashed_name = "test"
        cacher = Cacher(path, TensorType.pt)

        encoded_instance = {
            "input_ids": torch.rand(3),
            "tensors": torch.rand(3),
            "original_tokens": [[1, 2], [3]],
        }

        cacher.create_cache(hashed_name, encoded_instance)
        from_cache = cacher.get_from_cache(hashed_name)

        self.assertEqual(all(torch.eq(encoded_instance["input_ids"], from_cache["input_ids"])), True)
        self.assertEqual(all(torch.eq(encoded_instance["tensors"], from_cache["tensors"])), True)
        self.assertEqual(encoded_instance["original_tokens"], from_cache["original_tokens"])

        (path / hashed_name / str(TensorType.pt.value) / "input_ids.pt").unlink()
        (path / hashed_name / str(TensorType.pt.value) / "tensors.pt").unlink()
        (path / hashed_name / "original_tokens.txt").unlink()
        (path / hashed_name / str(TensorType.pt.value)).rmdir()
        (path / hashed_name).rmdir()
        path.rmdir()
