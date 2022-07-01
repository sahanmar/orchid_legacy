from config import Config, Context
from data_processing.coref_dataset import (
    get_dataset,
    CorefDataset,
    BucketBatchSampler
)
from nlp.evaluation import Evaluator
from nlp.models.common import set_seed
from nlp.models.torch.s2ecr import S2EModel
from nlp.training import Trainer
from utils.log import get_stream_logger
from utils.types import PipelineOutput, Response

logger = get_stream_logger('pipeline')


class OrchidPipeline:
    def __init__(self, config: Config = Config.from_path()):
        self.config = config
        self._init_pipeline_elements()

    def _init_pipeline_elements(self):
        # Added here for reproducibility
        set_seed(self.config.model.training.seed, n_gpu=Context.n_gpu)

        # Load Data
        logger.info('Starting data preparation')
        # Trainer
        if self.config.model.train:
            train_dataset: CorefDataset = get_dataset(
                'dev' if self.config.model.dev_mode else 'train',
                config=self.config
            )
            self.train_dataloader = BucketBatchSampler(
                train_dataset,
                batch_size_1=False
            )
            logger.info('Initializing the trainer')
            self.trainer = Trainer(config=self.config)
        else:
            self.trainer = None
            self.train_dataloader = None
        # Evaluator
        if self.config.model.eval:
            logger.info('Initializing the evaluator')
            self.evaluator = Evaluator(config=self.config)
        else:
            logger.info('Skipping the evaluator initialization')
            self.evaluator = None

        # Model
        logger.info('Initializing the model')
        self.model = S2EModel.from_config(config=self.config)
        logger.info(f'Putting the model to device={Context.device}')
        self.model.to(Context.device)

    def __call__(self):
        try:
            # Training
            if self.trainer is not None:
                self.trainer.run(
                    model=self.model,
                    batched_data=self.train_dataloader,
                    evaluator=self.evaluator
                )
            else:
                logger.info(f'Skipping the training step')

            # Evaluation
            if self.evaluator is not None:
                self.evaluator.evaluate(
                    model=self.model,
                    prefix='final'
                )
            logger.info('Finished')
            return PipelineOutput(state=Response.success)

        except Exception as ex:  # must specify the error type
            logger.error('Pipeline stopped with an exception', exc_info=ex)
            return PipelineOutput(state=Response.fail)


if __name__ == '__main__':
    pipeline = OrchidPipeline()
    pipeline()
