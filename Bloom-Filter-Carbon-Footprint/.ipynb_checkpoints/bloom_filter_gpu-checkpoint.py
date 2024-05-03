import numpy as np
import argparse
import pandas as pd
from random import randint
import mmh3 as murmurhash3_32  
from numba import cuda
import hashlib

def consistent_hash(url):
    return int(hashlib.sha256(url.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)

@cuda.jit
def test_kernel(table, k, hash_funcs_seeds, keys, result, m):
    idx = cuda.grid(1)
    if idx < len(keys):
        match = 0
        key = keys[idx]
        for j in range(k):
            seed = hash_funcs_seeds[j]
            t = (key * seed + 12345) % m
            match += table[t] == 1
        result[idx] = (match == k)

class BloomFilter_gpu():
    def __init__(self, n, hash_len):
        self.n = n
        self.hash_len = hash_len
        self.k = max(1, int(hash_len / n * 0.6931472)) if n > 0 else 1
        self.h_seeds = np.array([randint(1, 99999999) for _ in range(self.k)], dtype=np.int32)
        self.table = np.zeros(self.hash_len, dtype=np.uint32)

    def insert(self, keys):
        keys = np.array([consistent_hash(key) for key in keys], dtype=np.int32)
        threads_per_block = 256
        blocks_per_grid = (len(keys) + threads_per_block - 1) // threads_per_block
        insert_kernel[blocks_per_grid, threads_per_block](self.table, self.k, self.h_seeds, keys, self.hash_len)

    def test(self, keys, single_key=True):
        keys = np.array([consistent_hash(key) for key in keys], dtype=np.int32) if not single_key else np.array([consistent_hash(keys)], dtype=np.int32)
        result = np.zeros(len(keys), dtype=np.uint8)
        if self.hash_len > 0:
            threads_per_block = 256
            blocks_per_grid = (len(keys) + threads_per_block - 1) // threads_per_block
            test_kernel[blocks_per_grid, threads_per_block](self.table, self.k, self.h_seeds, keys, result, self.hash_len)
        if single_key:
            return result[0]  
        return result 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action="store", dest="data_path", type=str, required=True, help="path of the dataset")
    parser.add_argument('--size_of_Ada_BF', action="store", dest="R_sum", type=int, required=True, help="size of the Ada-BF")

    results = parser.parse_args()
    DATA_PATH = results.data_path
    R_sum = results.R_sum

    data = pd.read_csv(DATA_PATH)

    negative_sample = data.loc[(data['label'] == -1)]
    positive_sample = data.loc[(data['label'] == 1)]

    url = positive_sample['url'].to_numpy()
    n = len(url)
    bloom_filter = BloomFilter_gpu(n, R_sum)
    bloom_filter.insert(url)

    url_negative = negative_sample['url'].to_numpy()
    n1 = bloom_filter.test(url_negative, single_key=False)
    print('False positive items: ', sum(n1))
