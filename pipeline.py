from config import Config, Context
from data_processing.coref_dataset import (
    get_dataset,
    CorefDataset,
    BucketBatchSampler
)
from nlp.evaluation import Evaluator
from nlp.models.torch.s2ecr import S2EModel
from nlp.models.torch.s2ecr_training import Trainer
from utils.log import get_stream_logger
from utils.types import PipelineOutput, Response

logger = get_stream_logger(__name__)


class OrchidPipeline:
    def __init__(self, config: Config):
        self.config = config

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
            logger.info('Initializing the model')
            model = S2EModel.from_config(config=self.config)
            logger.info(f'Putting the model to device={Context.device}')
            model.to(Context.device)

            # Evaluator
            evaluator = Evaluator(config=self.config)

            # Trainer
            logger.info('Initializing the trainer')
            trainer = Trainer(config=self.config.model.training)
            trainer.train(
                model=model,
                batched_data=train_dataloader,
                evaluator=evaluator
            )

            logger.info('Finished')
            return PipelineOutput(state=Response.success)

        except Exception as ex:  # must specify the error type
            logger.error('Pipeline stopped with an exception', exc_info=ex)
            return PipelineOutput(state=Response.fail)
