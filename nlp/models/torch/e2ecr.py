import torch
import torch.nn as nn
import torch.optim as optim
import logging

from typing import Any, List, Tuple, Union, Dict

from utils.util_types import TokenRange

from datetime import datetime
from tqdm import tqdm

from pathlib import Path

CONTEXT = {
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "dtype": torch.float32,
}


class Score(nn.Module):
    """Generic scoring module"""

    def __init__(self, embeds_dim, hidden_dim=150):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x.to(CONTEXT["device"]))


class MentionScore(nn.Module):
    """Mention scoring module"""

    def __init__(self, gi_dim: int, attn_dim: int):
        super().__init__()

        self.gi_dim = gi_dim
        self.attention = Score(attn_dim).to(CONTEXT["device"])  # type: ignore
        self.score = Score(gi_dim).to(CONTEXT["device"])  # type: ignore

    def forward(self, batch_embeds: torch.Tensor, batch_spans_ids: List[List[TokenRange]], K=250):
        """
        Compute unary mention score for each span

        Input: BATCH x WORD_TOKENS x EMBED,
        Output: Tuple[BATCH x DOCUMENT_SPANS x 3*EMBED, BATCH x DOCUMENT_SPANS x 1]
        """

        # Compute attention for every doc token
        attns = self.attention(batch_embeds)

        # Get dimensions for BATCH x DOCUMENT_SPANS x 3*EMBED structure
        the_highest_span_count = max(len(document_span_ids) for document_span_ids in batch_spans_ids)

        # Create and fill BATCH x DOCUMENT_SPANS x 3*EMBED tensor
        batch_document_span_embeds = torch.stack(
            [
                torch.stack(
                    [
                        torch.cat(
                            [
                                batch_embeds[doc_id, span.start, :],  # First span token
                                batch_embeds[doc_id, span.end, :],  # Last span token
                                torch.sum(
                                    torch.mul(
                                        batch_embeds[doc_id, span.to_consecutive_list(), :],
                                        attns[doc_id, span.to_consecutive_list(), :],
                                    ),
                                    dim=0,
                                ),  # Attns through spans
                            ]
                        )
                        for span in document_span_ids
                    ]
                    + [
                        torch.zeros((self.gi_dim), requires_grad=True).to(CONTEXT["device"])
                        for _ in range(max(the_highest_span_count - len(document_span_ids), 0))
                    ]
                )
                for doc_id, document_span_ids in enumerate(batch_spans_ids)
            ]
        )

        # Compute each span's unary mention score
        mention_scores = self.score(batch_document_span_embeds)

        return batch_document_span_embeds, mention_scores


class PairwiseScore(nn.Module):
    """Coreference pair scoring module"""

    def __init__(self, gij_dim):
        super().__init__()

        self.score = Score(gij_dim)

    def forward(self, batch_document_span_embeds, mention_scores):
        """
        Compute pairwise score for spans and their up to K antecedents

        Input: Tuple[BATCH x DOCUMENT_SPANS x 3*EMBED, BATCH x DOCUMENT_SPANS x 1]
        Output: BATCH x DOCUMENT_SPANS x DOCUMENT_SPANS * 1
        """

        # Create pairs of spans
        batch_document_span_pairs_embeds = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack([torch.cat((span_1, span_2, span_1 * span_2)) for span_2 in document])
                        for span_1 in document
                    ]
                )
                for document in batch_document_span_embeds
            ]
        )

        # # Score span pairs as coref
        span_pairs_scores = self.score(batch_document_span_pairs_embeds)

        # Stack mention and span cores scores
        span_pairs_extended_scores = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [
                                torch.mean(
                                    torch.stack(
                                        [span_i_mention_scores, span_j_mention_scores, span_ij_pair_scores]
                                    )
                                )
                                for span_j_mention_scores, span_ij_pair_scores in zip(
                                    doc_mention_scores, span_i_pair_scores
                                )
                            ]
                        )
                        for span_i_mention_scores, span_i_pair_scores in zip(
                            doc_mention_scores, doc_pair_scores
                        )
                    ]
                )
                for doc_mention_scores, doc_pair_scores in zip(mention_scores, span_pairs_scores)
            ]
        )

        return span_pairs_extended_scores


class E2ECR(nn.Module):
    def __init__(self, embeds_dim: int, hidden_dim: int, distance_dim: int = 20, speaker_dim: int = 20):
        super().__init__()

        # Forward and backward pass over the document
        attn_dim = embeds_dim

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = 3 * embeds_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim * 3

        self.score_spans = MentionScore(gi_dim, attn_dim).to(CONTEXT["device"])  # type: ignore
        self.score_pairs = PairwiseScore(gij_dim).to(CONTEXT["device"])  # type: ignore

    def forward(self, encoded_docs: torch.Tensor, spans_ids: List[List[TokenRange]]) -> torch.Tensor:
        """
        Predict pairwise coreference scores
        """

        # Get mention scores for each span
        g_i, mention_scores = self.score_spans(encoded_docs, spans_ids)

        # Get pairwise scores for each span combo
        coref_scores = self.score_pairs(g_i, mention_scores)

        return torch.clamp(coref_scores, min=0, max=1)


class Trainer:
    """Class dedicated to training and evaluating the model"""

    def __init__(self, model: E2ECR, lr: float = 1e-3, enable_cuda=False):

        self.model = to_cuda(model) if enable_cuda else model

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.001)

        self.loss_fn = nn.BCELoss()

        self.logger = training_logger()

    def train(
        self,
        train_data: Tuple[List[torch.Tensor], List[List[List[TokenRange]]], List[torch.Tensor]],
        test_data: Tuple[List[torch.Tensor], List[List[List[TokenRange]]], List[torch.Tensor]],
        folder_to_save: Path,
        num_epochs: int,
    ):
        """ Train a model """

        best_f1 = -1.0

        loss_evolution: List[float] = []

        for epoch in range(1, num_epochs + 1):
            loss, average_f1 = self.train_epoch(train_data, test_data, epoch)
            loss_evolution.extend(loss)

            # Save the model with the highest training f1
            if average_f1 > best_f1:
                best_f1 = average_f1
                self.save_model(folder_to_save / "e2ecr_model.pt")

        with open("loss_evolution.txt", "w") as f:
            for l in loss_evolution:
                f.write("%s\n" % l)

    def train_epoch(
        self,
        train_data: Tuple[List[torch.Tensor], List[List[List[TokenRange]]], List[torch.Tensor]],
        test_data: Tuple[List[torch.Tensor], List[List[List[TokenRange]]], List[torch.Tensor]],
        epoch: int,
    ) -> Tuple[List[float], float]:
        """ Run a training epoch over 'steps' documents """

        # Set model to train (enables dropout)
        self.model.train()

        epoch_loss: List[float] = []
        train_f1: List[float] = []

        for train_instances, train_span_ids, train_target in tqdm(zip(*train_data)):
            # Compute loss, number gold links found, total gold links
            metrics = self.train_doc(
                train_instances.to(CONTEXT["device"]),
                train_span_ids,
                train_target.to(CONTEXT["device"]),
            )

            # Track stats by document for debugging
            training_stats = f" Epoch: {epoch} | F1 : {metrics['f1']} | Loss: {metrics['loss']} | Accuracy: {metrics['accuracy']} | Precision: {metrics['precision']} | Recall: {metrics['recall']}\n"
            print("\n")
            print("TRAINING")
            print(training_stats)
            self.logger.info("TRAINING")
            self.logger.info(training_stats)

            epoch_loss.append(metrics["loss"])
            train_f1.append(metrics["f1"])

        # Evaluate model
        self.model.eval()
        test_f1, test_acc, test_prec, test_recall = [], [], [], []
        for test_instances, test_span_ids, test_target in zip(*test_data):
            test_probs = self.model(test_instances.to(CONTEXT["device"]), test_span_ids)
            f1, acc, prec, recall = get_scores(test_probs, test_target.to(CONTEXT["device"]))
            test_f1.append(f1)
            test_acc.append(acc)
            test_prec.append(prec)
            test_recall.append(recall)

        testing_stats = f" Epoch: {epoch} | F1 : {safe_avg(test_f1)} | Accuracy: {safe_avg(test_acc)} | Precision: {safe_avg(test_prec)} | Recall: {safe_avg(test_recall)}\n"
        print("TESTING")
        print(testing_stats)
        self.logger.info("TESTING")
        self.logger.info(testing_stats)

        # Step the learning rate decrease scheduler
        # self.scheduler.step()
        return epoch_loss, sum(train_f1) / len(train_f1)

    def train_doc(
        self,
        instances: torch.Tensor,
        span_ids: List[List[TokenRange]],
        target: torch.Tensor,
    ) -> Dict[str, Any]:
        """ Compute loss for a forward pass over a document """

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Predict coref probabilites for each span in a document
        probs = self.model(instances, span_ids)

        # Cross entropy log-likelihood
        loss = self.loss_fn(probs.view(-1).to(CONTEXT["device"]), target.view(-1))

        accuracy, precision, recall, f1 = get_scores(probs, target)

        # Backpropagate
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        return {"loss": loss.item(), "f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall}

    def save_model(self, savepath: Path) -> None:
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)


def get_scores(probs: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float]:
    # Naive accuracy result
    thresholded_probs = (probs > 0.5).float()
    accuracy = (thresholded_probs == target.squeeze(-1)).float().sum() / torch.numel(probs)

    # Naive precision result
    tp = (thresholded_probs * target.squeeze(-1)).sum()
    fp = (thresholded_probs * (target.squeeze(-1) == 0).float()).sum()
    precision = 0 if tp == 0 and fp == 0 else tp / (tp + fp)

    # Naive recall result
    fn = ((probs <= 0.5).float() * target.squeeze(-1)).sum().float().sum()
    recall = 0 if tp == 0 and fn == 0 else tp / (tp + fn)

    # Naive F1
    f1 = tp / (tp + 1 / 2 * (fp + fn))

    return accuracy, precision, recall, f1


def safe_avg(data: List[float]) -> float:
    if not data:
        return 0.0
    return sum(data) / len(data)


def to_cuda(model: Union[nn.Module, torch.Tensor]):
    return model.to(torch.device("cuda"))


def create_target_values(
    texts_spans: List[List[TokenRange]], text_correfs: List[Dict[int, List[TokenRange]]]
) -> torch.Tensor:
    batch_size = len(texts_spans)
    max_spans_num = max((len(spans) for spans in texts_spans))
    targer_values = torch.zeros((batch_size, max_spans_num, max_spans_num, 1))
    for k, spans in enumerate(texts_spans):
        for i, span_i in enumerate(spans):
            intersection_labels = [
                label
                for label, correfs in text_correfs[k].items()
                for corref in correfs
                if corref.inside(span_i)
            ]
            if intersection_labels:
                for intersection_label in intersection_labels:
                    for j, span_j in enumerate(spans):
                        if any([corref.inside(span_j) for corref in text_correfs[k][intersection_label]]):
                            targer_values[k, i, j, :] = 1

    return targer_values


def training_logger() -> logging.Logger:
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename="app.log", mode="a")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    return logger
