import torch
import torch.nn as nn
import torch.optim as optim

from typing import List, Tuple, Union, Dict

from torch import Tensor
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

        batch_size, document_spans_size, embed_size = list(batch_document_span_embeds.size())

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

    def train(
        self,
        train_data: Tuple[List[torch.Tensor], List[List[List[TokenRange]]], List[torch.Tensor]],
        test_data: Tuple[List[torch.Tensor], List[List[List[TokenRange]]], List[torch.Tensor]],
        folder_to_save: Path,
        num_epochs: int,
        eval_interval: int = 10,
    ):
        """ Train a model """
        for epoch in range(1, num_epochs + 1):
            _, _ = self.train_epoch(train_data, epoch)

            # Save often
            self.save_model(folder_to_save / (str(datetime.now()) + ".pt"))

            # TODO Implemend eval step

            # Evaluate every eval_interval epochs
            # if epoch % eval_interval == 0:
            #     print("\n\nEVALUATION\n\n")
            #     self.model.eval()
            #     results = self.evaluate(self.val_corpus)
            #     print(results)

    def train_epoch(
        self,
        data: Tuple[List[torch.Tensor], List[List[List[TokenRange]]], List[torch.Tensor]],
        epoch: int,
    ) -> Tuple[List[float], List[float]]:
        """ Run a training epoch over 'steps' documents """

        # Set model to train (enables dropout)
        self.model.train()

        epoch_loss, epoch_accuracy = [], []

        for instances, span_ids, target in tqdm(zip(*data)):

            # Compute loss, number gold links found, total gold links
            loss, accuracy, precision, recall = self.train_doc(
                instances.to(CONTEXT["device"]), span_ids, target.to(CONTEXT["device"])
            )

            # Track stats by document for debugging
            print(
                epoch,
                f"| Loss: {loss} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}",
            )
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)

        # Step the learning rate decrease scheduler
        # self.scheduler.step()
        return epoch_loss, epoch_accuracy

    def train_doc(
        self, instances: torch.Tensor, span_ids: List[List[TokenRange]], target: torch.Tensor
    ) -> Tuple[float, float, float, float]:
        """ Compute loss for a forward pass over a document """

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Predict coref probabilites for each span in a document
        probs = self.model(instances, span_ids)

        # Cross entropy log-likelihood
        loss = self.loss_fn(probs.view(-1).to(CONTEXT["device"]), target.view(-1))

        # Naive accuracy result
        thresholded_probs = (probs > 0.5).float()
        accuracy = (thresholded_probs == target.squeeze(-1)).float().sum() / torch.numel(probs)

        # Naive precision result
        tp = (thresholded_probs * target.squeeze(-1)).sum()
        fp = (thresholded_probs * (target.squeeze(-1) == 0).float()).sum()
        precision = 0 if tp == 0 and fp == 0 else tp / (tp + fp)

        # Naive recall result
        fn = ((probs <= 0.5).float() * target.squeeze(-1)).sum().float().sum()
        recall = fn / (tp + fn)

        # Backpropagate
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        return loss.item(), accuracy, precision, recall

    def save_model(self, savepath: Path) -> None:
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)


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
