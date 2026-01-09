import numpy as np


def calculate_rank(score, target, filter_list):
    score_target = score[target]
    score[filter_list] = score_target - 1
    rank = np.sum(score > score_target) + np.sum(score == score_target) // 2 + 1
    return rank


def metrics(rank):
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1


def get_rank(score, ent, filter_list):
    score = score.copy()
    return calculate_rank(score, ent, filter_list)


def get_topK(score, ent, filter_list, topK):
    score = score.copy()
    target_score = score[ent]
    score[filter_list] = target_score - 1
    score[ent] = target_score
    indices = np.argsort(-score, kind='stable')  # 负号表示降序
    vals = score[indices]
    return indices[:topK], vals[:topK]
