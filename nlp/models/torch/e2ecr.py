import torch
import torch.nn as nn


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
        return self.score(x)


class MentionScore(nn.Module):
    """Mention scoring module"""

    def __init__(self, gi_dim, attn_dim):
        super().__init__()

        self.attention = Score(attn_dim)
        self.score = Score(gi_dim)

    def forward(self, embeds, doc, K=250):
        """
        Compute unary mention score for each span

        Input: BATCH x PADDED_SENTS x EMBED
        Output: Tuple[ BATCH x PADDED_SENTS x 2*EMBED, BATCH x PADDED_SENTS x 1]
        """

        # Compute first part of attention over span states (alpha_t)
        attns = self.attention(embeds)

        # Weight attention values using softmax
        attn_weights = F.softmax(attns, dim=1)

        # Compute self-attention over embeddings (x_hat)
        attn_embeds = torch.sum(torch.mul(embeds, attn_weights), dim=1)

        # Cat it all together to get g_i, our span representation
        g_i = torch.cat((embeds, attn_embeds), dim=1)

        # Compute each span's unary mention score
        mention_scores = self.score(g_i)

        return g_i, mention_scores


class PairwiseScore(nn.Module):
    """Coreference pair scoring module"""

    def __init__(self, gij_dim, distance_dim, speaker_dim):
        super().__init__()

        self.distance = Distance(distance_dim)
        self.speaker = Speaker(speaker_dim)

        self.score = Score(gij_dim)

    def forward(self, g_i, mention_scores):
        """Compute pairwise score for spans and their up to K antecedents"""

        # Extract raw features
        mention_ids, antecedent_ids, distances, genres, speakers = zip(
            *[(i.id, j.id, i.i2 - j.i1, i.genre, speaker_label(i, j)) for i in spans for j in i.yi]
        )

        # For indexing a tensor efficiently
        mention_ids = to_cuda(torch.tensor(mention_ids))
        antecedent_ids = to_cuda(torch.tensor(antecedent_ids))

        # Embed them
        phi = torch.cat((self.distance(distances), self.genre(genres), self.speaker(speakers)), dim=1)

        # Extract their span representations from the g_i matrix
        i_g = torch.index_select(g_i, 0, mention_ids)
        j_g = torch.index_select(g_i, 0, antecedent_ids)

        # Create s_ij representations
        pairs = torch.cat((i_g, j_g, i_g * j_g, phi), dim=1)

        # Extract mention score for each mention and its antecedents
        s_i = torch.index_select(mention_scores, 0, mention_ids)
        s_j = torch.index_select(mention_scores, 0, antecedent_ids)

        # Score pairs of spans for coreference link
        s_ij = self.score(pairs)

        # Compute pairwise scores for coreference links between each mention and
        # its antecedents
        coref_scores = torch.sum(torch.cat((s_i, s_j, s_ij), dim=1), dim=1, keepdim=True)

        # Update spans with set of possible antecedents' indices, scores
        spans = [
            attr.evolve(span, yi_idx=[((y.i1, y.i2), (span.i1, span.i2)) for y in span.yi])
            for span, score, (i1, i2) in zip(spans, coref_scores, pairwise_indexes(spans))
        ]

        # Get antecedent indexes for each span
        antecedent_idx = [len(s.yi) for s in spans if len(s.yi)]

        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores = [to_cuda(torch.tensor([]))] + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(torch.tensor([[0.0]]))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        # Batch and softmax
        # get the softmax of the scores for each span in the document given
        probs = [F.softmax(tensr) for tensr in with_epsilon]

        # pad the scores for each one with a dummy value, 1000 so that the tensors can
        # be of the same dimension for calculation loss and what not.
        probs, _ = pad_and_stack(probs, value=1000)
        probs = probs.squeeze()

        return spans, probs


class E2ECR(nn.Module):
    def __init__(self, embeds_dim, hidden_dim, distance_dim=20, speaker_dim=20):
        super().__init__()

        # Forward and backward pass over the document
        attn_dim = hidden_dim * 2

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = embeds_dim + distance_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim * 3

        self.score_spans = MentionScore(gi_dim, attn_dim, distance_dim)
        self.score_pairs = PairwiseScore(gij_dim, distance_dim, speaker_dim)

    def forward(self, encoded_doc):
        """
        Predict pairwise coreference scores
        """

        # Get mention scores for each span, prune
        g_i, mention_scores = self.score_spans(encoded_doc)

        # Get pairwise scores for each span combo
        coref_scores = self.score_pairs(g_i, mention_scores)

        return spans, coref_scores


# # print('Initializing...')
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchtext.vocab import Vectors

# import random
# import numpy as np
# import networkx as nx
# from tqdm import tqdm
# from random import sample
# from datetime import datetime
# from subprocess import Popen, PIPE
# from boltons.iterutils import pairwise
# from loader import *
# from utils import *
