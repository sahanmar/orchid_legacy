from typing import Optional

from config import Config, Context
from data_processing.cacher import Cacher
from data_processing.coref_dataset import (
    get_dataset,
    CorefDataset,
    BucketBatchSampler
)
from nlp.models.torch.s2ecr import S2EModel
from nlp.models.torch.s2ecr_training import Trainer
from utils.log import get_stream_logger
from utils.types import PipelineOutput, Response

logger = get_stream_logger(__name__)


class OrchidPipeline:
    def __init__(
            self,
            config: Config,
            cacher: Optional[Cacher] = None,
    ):
        self.config = config

        if cacher is None:
            self.cacher = Cacher.from_config(self.config.cache) \
                if self.config.cache is not None else None
        else:
            self.cacher = cacher

    def __call__(self):
        try:
            # Load Data
            logger.info('Starting data preparation')
            train_dataset: CorefDataset = get_dataset('dev', config=self.config)
            train_dataloader: BucketBatchSampler = BucketBatchSampler(
                train_dataset,
                max_seq_len=self.config.encoding.max_seq_len,
                max_total_seq_len=self.config.text.max_total_seq_len,
                batch_size_1=self.config.model.params.batch_size == 1
            )

            # Model
            logger.info('Starting model preparation')
            model = S2EModel.from_config(config=self.config).to(Context.device)
            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     model = torch.nn.DataParallel(model)

            # Trainer
            logger.info('Initializing Trainer')
            trainer = Trainer(config=self.config.model.training)
            trainer.train(model=model, batched_data=train_dataloader)

            return PipelineOutput(state=Response.success)

        except Exception as ex:  # must specify the error type
            logger.error('Pipeline stopped with an exception', exc_info=ex)
            return PipelineOutput(state=Response.fail)
