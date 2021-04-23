import unittest

from pathlib import Path

from config.config import Config, DataPaths
from pipeline import OrchidPipeline
from utils.util_types import Response

config = Config.load_cfg(Path("test_stages/test_config.json"))


class PipelineTest(unittest.TestCase):
    def test_pipeline_success_no_cache(self):
        pipeline = OrchidPipeline.from_config(config)
        pipeline_output = pipeline()
        self.assertEqual(pipeline_output.state, Response.success)

    def test_pipeline_fail_no_cache(self):
        fail_config = Config(
            data_path=DataPaths(Path(""), Path(""), Path("")), model=config.model, encoding=config.encoding
        )
        pipeline = OrchidPipeline.from_config(fail_config)
        pipeline_output = pipeline()
        self.assertEqual(pipeline_output.state, Response.fail)
