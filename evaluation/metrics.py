from typing import List, Set, Tuple


def f1_score(precision: float, recall: float) -> float:
    return 2 / (1 / precision + 1 / recall)


def muc(key: List[Set[int]], response: List[Set[int]]) -> Tuple[float, float, float]:

    # recall
    key_partions_lens: List[int] = []

    for k in key:
        response_coverage = [el for el in [len(k.intersection(r)) for r in response] if el > 0]
        singltones = len(k) - sum(response_coverage)
        key_partions_lens.append(len(response_coverage) - singltones)

    recall = sum((len(k) - kpl for k, kpl in zip(key, key_partions_lens))) / sum((len(k) - 1 for k in key))

    # precision
    response_partions_lens: List[int] = []

    for r in response:
        response_coverage = [el for el in [len(r.intersection(k)) for k in key] if el > 0]
        singltones = len(r) - sum(response_coverage)
        response_partions_lens.append(len(response_coverage) - singltones)

    precision = sum((len(r) - rpl for r, rpl in zip(response, response_partions_lens))) / sum(
        (len(r) - 1 for r in response)
    )

    return precision, recall, f1_score(precision, recall)


def b_cubed(key: List[Set[int]], response: List[Set[int]]) -> Tuple[float, float, float]:

    tp = 0
    rec_per_value = []
    prec_per_value = []

    for cluster in response:
        for control_value in cluster:
            for k in key:
                if control_value in k:
                    k_len = len(k)
                    for reference_value in cluster:
                        if reference_value in k:
                            tp += 1
                    break
            rec_per_value.append(tp / k_len)
            prec_per_value.append(tp / len(cluster))
            tp = 0
            k_len = 0

    recall = sum(rec_per_value) / len(rec_per_value)
    precision = sum(prec_per_value) / len(prec_per_value)

    return precision, recall, f1_score(precision, recall)
