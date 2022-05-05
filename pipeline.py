from config import Config, Context
from data_processing.coref_dataset import (
    get_dataset,
    CorefDataset,
    BucketBatchSampler
)
from nlp.evaluation import Evaluator
from nlp.models.torch.s2ecr import S2EModel
from nlp.training import Trainer
from utils.log import get_stream_logger
from utils.types import PipelineOutput, Response

logger = get_stream_logger('pipeline')


class OrchidPipeline:
    def __init__(self, config: Config = Config.from_path()):
        self.config = config

    def __call__(self):
        try:
            # Load Data
            logger.info('Starting data preparation')
            train_dataset: CorefDataset = get_dataset(
                'dev' if self.config.model.dev_mode else 'train',
                config=self.config
            )
            train_dataloader: BucketBatchSampler = BucketBatchSampler(
                train_dataset,
                max_seq_len=self.config.encoding.max_seq_len,
                max_total_seq_len=self.config.text.max_total_seq_len,
                batch_size_1=True
            )

            # Model
            logger.info('Initializing the model')
            model = S2EModel.from_config(config=self.config)
            logger.info(f'Putting the model to device={Context.device}')
            model.to(Context.device)

            # Evaluator
            if self.config.model.eval:
                logger.info('Initializing the evaluator')
                evaluator = Evaluator(config=self.config)
            else:
                logger.info('Skipping the evaluator initialization')
                evaluator = None

            # Trainer
            if self.config.model.train:
                logger.info('Initializing the trainer')
                trainer = Trainer(config=self.config)
                trainer.run(
                    model=model,
                    batched_data=train_dataloader,
                    evaluator=evaluator
                )
            else:
                logger.info(f'Skipping the training step')

            logger.info('Finished')
            return PipelineOutput(state=Response.success)

        except Exception as ex:  # must specify the error type
            logger.error('Pipeline stopped with an exception', exc_info=ex)
            return PipelineOutput(state=Response.fail)


if __name__ == '__main__':
    pipeline = OrchidPipeline()
    pipeline()
