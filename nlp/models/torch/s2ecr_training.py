import json
import logging
import random
import numpy as np
import torch
from torch.functional import split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from coref_bucket_batch_sampler import BucketBatchSampler
from tqdm import tqdm, trange
from pathlib import Path

from transformers import AdamW, get_linear_schedule_with_warmup
from config.config import TrainingCgf
from nlp.models.torch.s2ecr import S2ECR

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        split_value: float,
        training_folder: Path,
        training_epochs: int,
        head_learning_rate: float,
        warmup_steps: int,
        amp: bool,
        local_rank: int,
    ):
        self.split_value = split_value
        self.training_folder = training_folder
        self.tb_path = Path(self.training_folder / "tensorboard")
        self.tb_writer = SummaryWriter(self.tb_path, flush_secs=30)
        self.training_epochs = training_epochs
        self.head_learning_rate = head_learning_rate
        self.warmup_steps = warmup_steps
        self.optimizer_path = Path(self.training_folder / "optimizer.pt")
        self.scheduler_path = Path(self.training_folder / "scheduler.pt")
        self.amp = amp
        self.local_rank = local_rank

        if not self.tb_path.is_dir():
            self.tb_path.mkdir()

    @staticmethod
    def from_config(self, config: TrainingCgf) -> "Trainer":
        return Trainer(
            split_value=config.split_value,
            training_folder=config.training_folder,
            training_epochs=config.training_epochs,
            head_learning_rate=config.head_learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            optimizer_path=config.optimizer_path,
            scheduler_path=config.scheduler_path,
            local_rank=config.local_rank,
        )

    def train(self, model: S2ECR, batched_data: DataLoader) -> None:
        """ Train the model """

        logger.info("Tensorboard summary path: %s" % self.tb_path)

        t_total = len(batched_data) * self.training_epochs

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

        head_learning_rate = self.head_learning_rate if self.head_learning_rate else self.learning_rate
        optimizer_grouped_parameters = [
            {"params": model_decay, "lr": self.learning_rate, "weight_decay": self.weight_decay},
            {"params": model_no_decay, "lr": self.learning_rate, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_learning_rate, "weight_decay": self.weight_decay},
            {"params": head_no_decay, "lr": head_learning_rate, "weight_decay": 0.0},
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=t_total
        )

        loaded_saved_optimizer = False
        # Check if saved optimizer or scheduler states exist

        if self.scheduler_path.is_file() and self.optimizer_path.is_file():
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(self.optimizer_path))
            scheduler.load_state_dict(torch.load(self.scheduler_path))
            loaded_saved_optimizer = True

        if self.amp:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16_opt_level)

        # Distributed training (should be after apex fp16 initialization)
        if self.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )


def train(args, train_dataset, model, tokenizer, evaluator, config: TrainingCgf):

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from global step %d", global_step)
            if not loaded_saved_optimizer:
                logger.warning(
                    "Training is continued from checkpoint, but didn't load optimizer and scheduler"
                )
        except ValueError:
            logger.info("  Starting fine-tuning.")
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    # If nonfreeze_params is not empty, keep all params that are
    # not in nonfreeze_params fixed.
    if args.nonfreeze_params:
        names = []
        for name, param in model.named_parameters():
            freeze = True
            for nonfreeze_p in args.nonfreeze_params.split(","):
                if nonfreeze_p in name:
                    freeze = False

            if freeze:
                param.requires_grad = False
            else:
                names.append(name)

        print("nonfreezing layers: {}".format(names))

    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproducibility
    set_seed(args)
    best_f1 = -1
    best_global_step = -1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(tensor.to(args.device) for tensor in batch)
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

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                losses = {key: val.mean() for key, val in losses.items()}
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    logger.info(f"\nloss step {global_step}: {(tr_loss - logging_loss) / args.logging_steps}")
                    tb_writer.add_scalar(
                        "Training_Loss", (tr_loss - logging_loss) / args.logging_steps, global_step
                    )
                    for key, value in losses.items():
                        logger.info(f"\n{key}: {value}")

                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.do_eval
                    and args.eval_steps > 0
                    and global_step % args.eval_steps == 0
                ):
                    results = evaluator.evaluate(
                        model, prefix=f"step_{global_step}", tb_writer=tb_writer, global_step=global_step
                    )
                    f1 = results["f1"]
                    if f1 > best_f1:
                        best_f1 = f1
                        best_global_step = global_step
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
                    logger.info(f"best f1 is {best_f1} on global step {best_global_step}")
                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                    and (not args.save_if_best or (best_global_step == global_step))
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if 0 < t_total < global_step:
            train_iterator.close()
            break

    with open(os.path.join(args.output_dir, f"best_f1.json"), "w") as f:
        json.dump({"best_f1": best_f1, "best_global_step": best_global_step}, f)

    tb_writer.close()
    return global_step, tr_loss / global_step


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
