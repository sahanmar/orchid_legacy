import json
import random
from pathlib import Path
from typing import (
    Tuple,
    Optional
)

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from config import Context, Config
from utils.log import get_stream_logger
from nlp.models.torch.s2ecr import S2EModel
from nlp.evaluation import Evaluator

logger = get_stream_logger('s2e-training')


class Trainer:
    def __init__(
            self,
            config: Config
    ):
        self.config = config.model.training
        self.do_eval = config.model.eval
        self.tb_path = Path(self.config.training_folder).joinpath("tensorboard")
        self.tb_writer = SummaryWriter(str(self.tb_path), flush_secs=30)
        self.optimizer_path = Path(self.config.training_folder).joinpath("optimizer.pt")
        self.scheduler_path = Path(self.config.training_folder).joinpath("scheduler.pt")
        self.output_folder = Path(self.config.training_folder).joinpath("output")

        if not self.tb_path.is_dir():
            self.tb_path.mkdir()
        if not self.output_folder.is_dir():
            self.output_folder.mkdir()

        # Losses
        self.tr_loss = 0.
        self.logging_loss = 0.

        # Counters
        self.global_step = 0

    def run(
            self,
            model: S2EModel,
            batched_data: DataLoader,
            evaluator: Optional[Evaluator] = None,
    ) -> Tuple[float, float]:
        """ Train the model """

        logger.info("Tensorboard summary path: %s" % self.tb_path)
        t_total = \
            len(batched_data) // \
            self.config.gradient_accumulation_steps * \
            self.config.training_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        head_params = ["coref", "mention", "antecedent"]

        model_decay = [
            p
            for n, p in model.named_parameters()
            if not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
        ]
        model_no_decay = [
            p
            for n, p in model.named_parameters()
            if not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
        ]
        head_decay = [
            p
            for n, p in model.named_parameters()
            if any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)
        ]
        head_no_decay = [
            p
            for n, p in model.named_parameters()
            if any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)
        ]

        head_learning_rate = self.config.head_learning_rate \
            if self.config.head_learning_rate \
            else self.config.learning_rate
        optimizer_grouped_parameters = [
            {"params": model_decay, "lr": self.config.learning_rate, "weight_decay": self.config.weight_decay},
            {"params": model_no_decay, "lr": self.config.learning_rate, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_learning_rate, "weight_decay": self.config.weight_decay},
            {"params": head_no_decay, "lr": head_learning_rate, "weight_decay": 0.0},
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=t_total
        )

        loaded_saved_optimizer = False
        # Check if saved optimizer or scheduler states exist
        if self.scheduler_path.is_file() and self.optimizer_path.is_file():
            logger.info(f'Reading states from the checkpoint')
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(self.optimizer_path))
            scheduler.load_state_dict(torch.load(self.scheduler_path))
            loaded_saved_optimizer = True

        if self.config.amp:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.config.fp16_opt_level)

        # Multi-GPU setup
        if Context.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.config.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True,
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info("Number of examples = %d", sum([len(batch) for batch in batched_data]))
        logger.info("Number of epochs = %d", self.config.training_epochs)
        logger.info("Gradient accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info("Total optimization steps = %d", t_total)

        self.global_step = 0
        self.tr_loss, self.logging_loss = 0.0, 0.0
        model.zero_grad()
        # Added here for reproducibility
        set_seed(self.config.seed, n_gpu=Context.n_gpu)

        train_iterator = trange(
            0, int(self.config.training_epochs), desc="Epoch", disable=self.config.local_rank not in [-1, 0]
        )

        best_f1 = -1
        best_global_step = -1
        for _ in train_iterator:
            epoch_iterator = tqdm(batched_data, desc="Iteration", disable=self.config.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                # print(batch)Î
                batch = tuple(tensor.to(Context.device) for tensor in batch[1])
                input_ids, attention_mask, gold_clusters = batch
                model.train()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gold_clusters=gold_clusters,
                    return_all_outputs=False,
                )
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                losses = outputs[-1]

                if Context.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    losses = {key: val.mean() for key, val in losses.items()}
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                if self.config.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.tr_loss += loss.item()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    self.global_step += 1

                    # Log metrics
                    if (
                            self.config.local_rank in [-1, 0]
                            and self.config.logging_steps > 0
                            and self.global_step % self.config.logging_steps == 0
                    ):
                        loss_to_write = (self.tr_loss - self.logging_loss) / self.config.logging_steps
                        logger.info(f"loss step {self.global_step}: {loss_to_write}")
                        self.tb_writer.add_scalar("Training_Loss", loss_to_write, self.global_step)
                        for key, value in losses.items():
                            logger.info(f"{key}: {value}")

                        self.logging_loss = self.tr_loss

                    if (
                            self.config.local_rank in [-1, 0]
                            and self.do_eval
                            and self.config.logging_steps > 0
                            and self.global_step % self.config.logging_steps == 0
                            and evaluator is not None
                    ):
                        results = evaluator.evaluate(
                            model=model,
                            prefix=f"step_{self.global_step}",
                            tb_writer=self.tb_writer,
                            global_step=self.global_step,
                        )
                        f1 = results["f1"]
                        if f1 > best_f1:
                            best_f1 = f1
                            best_global_step = self.global_step
                            # Save model checkpoint
                            output_dir = self.output_folder.joinpath(f"checkpoint-{self.global_step}")
                            if not output_dir.is_dir():
                                output_dir.mkdir()

                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)

                            logger.info("Saving model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), output_dir.joinpath("optimizer.pt"))
                            torch.save(scheduler.state_dict(), output_dir.joinpath("scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
                        logger.info(f"best f1 is {best_f1} on global step {best_global_step}")
        train_iterator.close()

        with open(Path(self.config.training_folder).joinpath(f"best_f1.json"), "w") as f:
            json.dump({"best_f1": best_f1, "best_global_step": best_global_step}, f)

        self.tb_writer.close()
        return self.global_step, self.tr_loss / self.global_step


def set_seed(seed: int, n_gpu: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
