from typing import Tuple

import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers import (
    BertPreTrainedModel,
    LongformerModel,
    LongformerConfig,
)
from transformers.models.bert.modeling_bert import ACT2FN

from config import ModelParameters, Config
from utils.log import get_stream_logger
from ..common import (
    extract_clusters,
    extract_mentions_to_predicted_clusters_from_clusters,
    mask_tensor
)

_logger = get_stream_logger(__name__)


class FullyConnectedLayer(Module):
    def __init__(self, config: LongformerConfig, input_dim: int, output_dim: int, dropout_prob: float):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class S2EModel(BertPreTrainedModel):

    @staticmethod
    def from_config(config: Config) -> 'S2EModel':
        encoder_config = LongformerConfig.from_pretrained(
            config.encoding.encoder_path.value
        )
        return S2EModel(
            config=encoder_config,
            args=config.model.params
        )

    def __init__(
            self,
            config: LongformerConfig,
            args: ModelParameters
    ):
        super().__init__(config)
        self.max_span_length = args.max_span_length
        self.top_lambda = args.top_lambda
        self.ffnn_size = args.ffnn_size
        self.do_mlps = self.ffnn_size > 0
        self.ffnn_size = self.ffnn_size if self.do_mlps else config.hidden_size
        self.normalise_loss = args.normalise_loss

        self.encoding_model = LongformerModel(config)
        self.encoding_model.resize_token_embeddings()
        # Freeze the encoding model parameters if necessary
        self.encoding_model.requires_grad_(args.trainable_embeddings)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size,
                                                     args.dropout_prob) if self.do_mlps else None
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size,
                                                   args.dropout_prob) if self.do_mlps else None
        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size,
                                                   args.dropout_prob) if self.do_mlps else None
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size,
                                                 args.dropout_prob) if self.do_mlps else None

        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size,
                                                   args.dropout_prob) if self.do_mlps else None
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size,
                                                 args.dropout_prob) if self.do_mlps else None

        self.mention_start_classifier = Linear(self.ffnn_size, 1)
        self.mention_end_classifier = Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.antecedent_s2s_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2s_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.init_weights()

        # Loss
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.bce_weight = args.bce_weight

    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        idx = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return torch.less(idx, len_expanded).int()

    def _prune_topk_mentions(self, mention_logits: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[
        torch.Tensor, ...]:
        """
        :param mention_logits: Shape [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        :return:
        """
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch.topk(
            mention_logits.view(batch_size, -1),
            dim=-1,
            k=max_k
        )  # [batch_size, max_k]
        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        topk_1d_indices = \
            (topk_1d_indices * span_mask) + (1 - span_mask) * (
                    (seq_length ** 2) - 1
            )  # We take different k for each example and mask the rest of the span with the largest index
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        topk_mention_start_ids = torch.div(
            sorted_topk_1d_indices,
            seq_length,
            rounding_mode='floor'
        )  # [batch_size, max_k]
        topk_mention_end_ids = torch.remainder(sorted_topk_1d_indices, seq_length)  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                             topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]

        topk_mention_logits = \
            topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]

        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_logits

    def _mask_antecedent_logits(self, antecedent_logits, span_mask):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.tril(torch.ones_like(antecedent_logits, dtype=self.dtype),
                                      diagonal=-1)  # [max_k, max_k]
        antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [max_k, max_k]
        antecedent_logits = mask_tensor(antecedent_logits, antecedents_mask)
        return antecedent_logits

    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        """
        :param span_starts: [batch_size, max_k]
        :param span_ends: [batch_size, max_k]
        :param all_clusters: [batch_size, max_cluster_size, max_clusters_num, 2]
        :return: [batch_size, max_k, max_k + 1] - [b, i, j] == 1 if i is antecedent of j
        """
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k + 1), device='cpu')
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, (starts, ends, gold_clusters) in enumerate(
                zip(span_starts.cpu().tolist(), span_ends.cpu().tolist(), all_clusters_cpu)):
            gold_clusters = extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_predicted_clusters_from_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for i, (start, end) in enumerate(zip(starts, ends)):
                if (start, end) not in gold_mentions:
                    continue
                for j, (a_start, a_end) in enumerate(list(zip(starts, ends))[:i]):
                    if (a_start, a_end) in mention_to_gold_clusters[(start, end)]:
                        new_cluster_labels[b, i, j] = 1
        new_cluster_labels = new_cluster_labels.to(self.device)  # [max_k, max_k + 1]
        no_antecedents = 1 - torch.sum(new_cluster_labels, dim=-1).bool().float()  # [max_k]
        new_cluster_labels[:, :, -1] = no_antecedents
        return new_cluster_labels

    def _get_loss(self, coref_logits, cluster_labels_after_pruning, span_mask):
        """
        :param coref_logits: [batch_size, max_k, max_k]
        :param cluster_labels_after_pruning: [batch_size, max_k, max_k]
        :param span_mask: [batch_size, max_k]
        :return:
        """
        gold_coref_logits = mask_tensor(coref_logits, cluster_labels_after_pruning)

        gold_log_sum_exp = torch.logsumexp(gold_coref_logits, dim=-1)  # [batch_size, max_k]
        all_log_sum_exp = torch.logsumexp(coref_logits, dim=-1)  # [batch_size, max_k]

        losses = all_log_sum_exp - gold_log_sum_exp
        losses = losses * span_mask
        per_example_loss = torch.sum(losses, dim=-1)  # [batch_size]
        if self.normalise_loss:
            per_example_loss = per_example_loss / losses.size(-1)
        ml_loss = per_example_loss.mean()
        # Add the BCE factor, clamping the inputs for numerical stability
        loss_bce = self.bce_weight * self.bce(
            torch.clamp(coref_logits, -50, 50),
            cluster_labels_after_pruning
        )
        _zero = torch.tensor(0.).to(self.device)
        if torch.isclose(ml_loss, _zero) and \
                torch.allclose(gold_coref_logits[:, :, :-1], _zero):
            _logger.warning(
                f'Wow, the likelihood loss is 0, '
                f'but the model learnt no coreference clusters at all :/'
            )
        return ml_loss, loss_bce

    def _get_mention_mask(self, mention_logits_or_weights: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of size [batch_size, seq_length, seq_length] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits_or_weights: Either the span mention logits or weights, size [batch_size, seq_length, seq_length]
        """
        mention_mask = torch.ones_like(mention_logits_or_weights, dtype=self.dtype)
        mention_mask = torch.triu(mention_mask, diagonal=0)  # Upper triangular part
        mention_mask = torch.tril(mention_mask, diagonal=self.max_span_length - 1)  # Lower triangular part
        return mention_mask

    def _calc_mention_logits(self, start_mention_reps: torch.Tensor, end_mention_reps: torch.Tensor) -> torch.Tensor:
        start_mention_logits = self.mention_start_classifier(start_mention_reps).squeeze(-1)  # [seq_length]
        end_mention_logits = self.mention_end_classifier(end_mention_reps).squeeze(-1)  # [seq_length]

        temp = self.mention_s2e_classifier(start_mention_reps)  # [seq_length, ffnn_size]
        joint_mention_logits = torch.matmul(
            temp, end_mention_reps.permute([0, 2, 1]))  # [seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(
            -2)  # [seq_len, seq_len]
        mention_mask = self._get_mention_mask(mention_logits)  # [batch_size, seq_length, seq_length]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_length, seq_length]
        return mention_logits

    def _calc_coref_logits(self, top_k_start_coref_reps: torch.Tensor,
                           top_k_end_coref_reps: torch.Tensor) -> torch.Tensor:
        # s2s
        temp = self.antecedent_s2s_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2s_coref_logits = torch.matmul(
            temp,
            top_k_start_coref_reps.permute([0, 2, 1])
        )  # [batch_size, max_k, max_k]

        # e2e
        temp = self.antecedent_e2e_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2e_coref_logits = torch.matmul(
            temp,
            top_k_end_coref_reps.permute([0, 2, 1])
        )  # [batch_size, max_k, max_k]

        # s2e
        temp = self.antecedent_s2e_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2e_coref_logits = torch.matmul(
            temp,
            top_k_end_coref_reps.permute([0, 2, 1])
        )  # [batch_size, max_k, max_k]

        # e2s
        temp = self.antecedent_e2s_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2s_coref_logits = torch.matmul(
            temp,
            top_k_start_coref_reps.permute([0, 2, 1])
        )  # [batch_size, max_k, max_k]

        # sum all terms
        # [batch_size, max_k, max_k]
        coref_logits = top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits
        return coref_logits

    def forward(self, input_ids, attention_mask=None, gold_clusters=None, return_all_outputs=False):
        outputs = self.encoding_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, dim]

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        end_mention_reps = self.end_mention_mlp(sequence_output) if self.do_mlps else sequence_output

        start_coref_reps = self.start_coref_mlp(
            sequence_output) if self.do_mlps else sequence_output  # [seq_len, ffnn_size | dim]
        end_coref_reps = self.end_coref_mlp(
            sequence_output) if self.do_mlps else sequence_output  # [seq_len, ffnn_size | dim]

        # mention scores
        mention_logits = self._calc_mention_logits(start_mention_reps, end_mention_reps)

        # prune mentions
        mention_start_ids, mention_end_ids, span_mask, topk_mention_logits = \
            self._prune_topk_mentions(
                mention_logits,
                attention_mask
            )

        batch_size, _, dim = start_coref_reps.size()
        max_k = mention_start_ids.size(-1)
        size = (batch_size, max_k, dim)

        # Antecedent scores
        # gather reps
        topk_start_coref_reps = torch.gather(
            start_coref_reps, dim=1,
            index=mention_start_ids.unsqueeze(-1).expand(size)
        )  # [max_k, dim]
        topk_end_coref_reps = torch.gather(
            end_coref_reps, dim=1,
            index=mention_end_ids.unsqueeze(-1).expand(size)
        )  # [max_k, dim]
        coref_logits = self._calc_coref_logits(topk_start_coref_reps, topk_end_coref_reps)  # [max_k, max_k]

        final_logits = topk_mention_logits + coref_logits  # [max_k, max_k]
        final_logits = self._mask_antecedent_logits(final_logits, span_mask)  # [max_k, max_k]
        # Adding zero logits for null span
        final_logits = torch.cat(
            (
                final_logits,
                torch.zeros(
                    (batch_size, max_k, 1),
                    device=self.device
                )
            ),
            dim=-1
        )  # [max_k, max_k + 1]

        if return_all_outputs:
            outputs = (mention_start_ids, mention_end_ids, final_logits, mention_logits)
        else:
            outputs = tuple()

        if gold_clusters is not None:
            losses = {}
            labels_after_pruning = self._get_cluster_labels_after_pruning(mention_start_ids, mention_end_ids,
                                                                          gold_clusters)
            loss_ml, loss_bce = self._get_loss(final_logits, labels_after_pruning, span_mask)
            loss = loss_ml + loss_bce
            # loss = self.criterion(final_logits, labels_after_pruning)
            losses.update({"loss": loss, "loss_bce": loss_bce, "loss_ml": loss_ml})
            outputs = (loss,) + outputs + (losses,)

        return outputs
