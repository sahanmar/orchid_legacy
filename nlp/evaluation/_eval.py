import json
import os
from collections import OrderedDict, defaultdict
from typing import (
    Dict,
    Any
)

import numpy as np
import torch
from tqdm.auto import tqdm

from config import Config, Context
from data_processing.coref_dataset import (
    CorefDataset,
    BucketBatchSampler,
    get_dataset
)
from utils.log import get_stream_logger
from ._metrics import CorefEvaluator, MentionEvaluator
from ..models.common import (
    extract_clusters,
    extract_mentions_to_predicted_clusters_from_clusters,
    extract_clusters_for_decode
)

logger = get_stream_logger('evaluator')


class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        eval_dataset: CorefDataset = get_dataset(
            'dev' if self.config.model.dev_mode else 'test',
            config=self.config
        )
        self.eval_dataloader = BucketBatchSampler(
            eval_dataset,
            batch_size_1=True
        )

    def evaluate(
            self,
            model,
            prefix: str = "",
            tb_writer=None,
            global_step=None
    ) -> Dict[str, Any]:

        if self.config.model.training.evaluation_folder and \
                not os.path.exists(self.config.model.training.evaluation_folder) and \
                self.config.model.training.local_rank in [-1, 0]:
            os.makedirs(self.config.model.training.evaluation_folder)

        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

        # Evaluation
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("Number of examples: %d", len(self.eval_dataloader))
        model.eval()

        post_pruning_mention_evaluator = MentionEvaluator()
        mention_evaluator = MentionEvaluator()
        coref_evaluator = CorefEvaluator()
        losses = defaultdict(list)
        doc_to_prediction = {}
        doc_to_subtoken_map = {}
        dataloader_iter = tqdm(self.eval_dataloader, desc='Evaluation')
        for (doc_key, subtoken_maps), batch in dataloader_iter:

            batch = tuple(tensor.to(Context.device) for tensor in batch)
            input_ids, attention_mask, gold_clusters = batch

            with torch.no_grad():
                # print(input_ids.shape, attention_mask.shape)
                try:
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    gold_clusters=gold_clusters,
                                    return_all_outputs=True)
                except RuntimeError as ex:
                    logger.error(f'Could not process the following inputs:\n'
                                 f'input_ids={input_ids}\n'
                                 f'attention_mask={attention_mask}\n'
                                 f'gold_clusters={gold_clusters}',
                                 exc_info=ex)
                loss_dict = outputs[-1]
            dataloader_iter.set_postfix(
                {k: v.item() for k, v in loss_dict.items()}
            )

            if Context.n_gpu > 1:
                loss_dict = {key: val.mean() for key, val in loss_dict.items()}

            for key, val in loss_dict.items():
                losses[key].append(val.item())

            outputs = outputs[1:-1]

            batch_np = tuple(tensor.cpu().numpy() for tensor in batch)
            outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)
            for output in zip(*(batch_np + outputs_np)):
                gold_clusters = output[2]
                gold_clusters = extract_clusters(gold_clusters)
                mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
                gold_mentions = list(mention_to_gold_clusters.keys())

                (start_offsets,
                 end_offsets,
                 coref_logits,
                 mention_logits) = output[-4:]

                max_antecedents = np.argmax(coref_logits, axis=1)
                mention_to_antecedent = {
                    (
                        (int(start),
                         int(end)),
                        (int(start_offsets[max_antecedent]),
                         int(end_offsets[max_antecedent]))
                    )
                    for start, end, max_antecedent in
                    zip(start_offsets, end_offsets, max_antecedents)
                    if max_antecedent < len(start_offsets)
                }

                predicted_clusters, _ = extract_clusters_for_decode(mention_to_antecedent)
                candidate_mentions = list(zip(start_offsets, end_offsets))

                mention_to_predicted_clusters = extract_mentions_to_predicted_clusters_from_clusters(predicted_clusters)
                predicted_mentions = list(mention_to_predicted_clusters.keys())
                post_pruning_mention_evaluator.update(candidate_mentions, gold_mentions)
                mention_evaluator.update(predicted_mentions, gold_mentions)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters,
                                       mention_to_gold_clusters)
                doc_to_prediction[doc_key] = predicted_clusters
                doc_to_subtoken_map[doc_key] = subtoken_maps

        (post_pruning_mention_precision,
         post_pruning_mentions_recall,
         post_pruning_mention_f1) = post_pruning_mention_evaluator.get_prf()
        (mention_precision,
         mentions_recall,
         mention_f1) = mention_evaluator.get_prf()
        prec, rec, f1 = coref_evaluator.get_prf()

        results = [(key, sum(val) / len(val)) for key, val in losses.items()]
        results += [
            ("post-pruning mention precision", post_pruning_mention_precision),
            ("post-pruning mention recall", post_pruning_mentions_recall),
            ("post-pruning mention f1", post_pruning_mention_f1),
            ("mention precision", mention_precision),
            ("mention recall", mentions_recall),
            ("mention f1", mention_f1),
            ("precision", prec),
            ("recall", rec),
            ("f1", f1)
        ]
        logger.info("Evaluation Results: {}".format(prefix))
        for key, values in results:
            if isinstance(values, float):
                logger.info(f"{key:32} = {values:.3f}")
            else:
                logger.info(f"{key:32} = {values}")
            if tb_writer is not None and global_step is not None:
                tb_writer.add_scalar(key, values, global_step)

        if self.config.model.training.evaluation_folder:
            output_eval_file = os.path.join(
                self.config.model.training.evaluation_folder,
                "eval_results.txt"
            )
            with open(output_eval_file, "a") as writer:
                if prefix:
                    writer.write(f'\n{prefix}:\n')
                for key, values in results:
                    if isinstance(values, float):
                        writer.write(f"{key:32}={values:.3f}\n")
                    else:
                        writer.write(f"{key:32}={values}\n")

        results = OrderedDict(results)
        # TODO: fill in the experiment name from env
        results["experiment_name"] = "SOME_EXPERIMENT"
        results["data"] = prefix
        with open(
                os.path.join(
                    self.config.model.training.training_folder,
                    "results.jsonl"
                ),
                mode="a+"
        ) as f:
            f.write(json.dumps(results) + '\n')

        return results
