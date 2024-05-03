import numpy as np
import pandas as pd
import cupy as cp
from numba import cuda

class Ada_BloomFilter_gpu:
    def __init__(self, n, hash_len, k_max):
        self.n = n
        self.hash_len = hash_len
        self.table = cp.zeros(self.hash_len, dtype=cp.uint32)
        self.k_max = k_max
        self.seeds = cp.random.randint(1, 99999999, size=k_max, dtype=cp.int32)

    def insert(self, keys, k):
        for key in keys:
            for j in range(k):
                t = (key * self.seeds[j] + j) % self.hash_len
                self.table[t] += 1  # Increment to handle collisions

    def test(self, keys, k):
        results = cp.zeros(len(keys), dtype=cp.uint8)
        for idx, key in enumerate(keys):
            match = 0
            for j in range(k):
                t = (key * self.seeds[j] + j) % self.hash_len
                if self.table[t] > 0:
                    match += 1
            results[idx] = 1 if match == k else 0
        return results

def sort_scores_gpu(input_scores):
    return cp.sort(input_scores)

def set_thresholds_gpu(scores_sorted, thresholds, num_piece, c, k_min, k_max):
    for k in range(k_min, k_max):
        i = k - k_min
        index = max(0, len(scores_sorted) - int(num_piece * c ** i))
        thresholds[-(i + 2)] = scores_sorted[index]

def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = cp.arange(c_min, c_max + 10**(-6), 0.1)
    FP_opt = len(train_negative)
    bloom_filter_opt, thresholds_opt, k_max_opt = None, None, None

    for k_max in range(num_group_min, num_group_max + 1):
        for c in c_set:
            tau = cp.sum(c ** cp.arange(0, k_max + 1))
            bloom_filter = Ada_BloomFilter_gpu(positive_sample.shape[0], R_sum, k_max)
            thresholds = cp.zeros(k_max + 1, dtype=cp.float32)
            thresholds[-1] = 1.1

            score = cp.array(train_negative[train_negative['score'] <= thresholds[-1]]['score'].values)
            scores_sorted = sort_scores_gpu(score)

            num_negative = len(score)
            num_piece = int(num_negative / tau) + 1
            set_thresholds_gpu(scores_sorted, thresholds, num_piece, c, 0, k_max)

            url = cp.array(positive_sample['url'].values)
            score = cp.array(positive_sample['score'].values)

            for score_s, url_s in zip(score, url):
                ix = cp.searchsorted(thresholds, score_s, side='right') - 1
                k = k_max - ix
                bloom_filter.insert(cp.array([url_s]), k)

            url_negative = cp.array(train_negative[train_negative['score'] < thresholds[-2]]['url'].values)
            test_results = bloom_filter.test(url_negative, k_max - cp.searchsorted(thresholds, train_negative[train_negative['score'] < thresholds[-2]]['score'].values, side='right'))

            FP_items = int(cp.sum(test_results)) + len(train_negative[train_negative['score'] >= thresholds[-2]])

            if FP_opt > FP_items:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds.get()  # Transfer back to host
                k_max_opt = k_max

            print(f'False positive items: {FP_items}, Number of groups: {k_max}, c = {c:.2f}')

    return bloom_filter_opt, thresholds_opt, k_max_opt

# Example usage, define your train_negative and positive_sample DataFrames
