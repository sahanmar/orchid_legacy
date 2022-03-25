import unittest
from pathlib import Path

from config.config import Config, DataPaths, env_config
from pipeline import OrchidPipeline
from utils.types import Response


class PipelineTest(unittest.TestCase):

    def setUp(self) -> None:
        assert env_config['ORCHID_ENV'] == 'test', \
            'Run the test in the \"test\" environment'
        _core_path = Path(__file__).resolve().parents[1]
        self.config = Config.from_path(
            _core_path.joinpath(env_config.get('CONFIG_PATH'))
        )

    def test_pipeline_success_no_cache(self):
        pipeline = OrchidPipeline.from_config(self.config)
        pipeline_output = pipeline()
        self.assertEqual(pipeline_output.state, Response.success)

    def test_pipeline_fail_no_cache(self):
        fail_config = Config(
            data_path=DataPaths(Path(""), Path(""), Path("")),
            model=self.config.model,
            encoding=self.config.encoding,
            cache=self.config.cache,
            text=self.config.text,
        )
        pipeline = OrchidPipeline.from_config(fail_config)
        pipeline_output = pipeline()
        self.assertEqual(pipeline_output.state, Response.fail)
