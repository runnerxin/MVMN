# =============================================
# @File     : evaluate.py
# @Software : PyCharm
# @Time     : 2019/7/2 15:06
# =============================================

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def get_hit_precision(label_score_sorted_pair, k=10):
    hit_precision = []
    for rk in range(k):
        if label_score_sorted_pair[rk][0] == 1:
            hit_precision.append((k - rk) / k)

    if len(hit_precision) == 0:
        return 0
    else:
        return np.mean(hit_precision)


def get_map(label_score_sorted_pair):
    map_values = []
    cnt = 0
    for rk in range(len(label_score_sorted_pair)):
        if label_score_sorted_pair[rk][0] == 1:
            cnt += 1
            map_values.append(cnt / (rk+1))

    return np.mean(map_values)


def get_recall_k(label_score_sorted_pair, k):

    count, total_label = 0, 0
    for rk in range(len(label_score_sorted_pair)):
        if label_score_sorted_pair[rk][0] == 1:
            total_label += 1
            if rk < k:
                count += 1
    if total_label != 0:
        return count / total_label
    else:
        return 0


def get_precision_k(label_score_sorted_pair, k):

    count = 0
    for rk in range(k):
        if label_score_sorted_pair[rk][0] == 1:
            count += 1
    return count / k
