import random

import numpy as np
import torch

from config.config import NULL_ID_FOR_COREF


def set_seed(seed: int, n_gpu: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def extract_clusters(gold_clusters):
    gold_clusters = [
        tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters
    ]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters


def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold


def extract_clusters_for_decode(mention_to_antecedent):
    mention_to_antecedent = sorted(mention_to_antecedent)
    mention_to_cluster = {}
    clusters = []
    for mention, antecedent in mention_to_antecedent:
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            clusters[cluster_idx].append(mention)
            mention_to_cluster[mention] = cluster_idx

        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])
    clusters = [tuple(cluster) for cluster in clusters]
    return clusters, mention_to_cluster


def mask_tensor(
        t: torch.Tensor,
        mask: torch.Tensor,
        mask_val: int = 10_000,
) -> torch.Tensor:
    assert mask_val > 0., 'Mask value must be a positive int'
    t = t + ((1. - mask.float()) * (-mask_val))
    t = torch.clamp(t, min=-mask_val, max=mask_val)
    return t


def batch_split_idx(array_size: int, split_value: float) -> int:
    return int(array_size * split_value)
