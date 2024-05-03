import numpy as np
import argparse
import pandas as pd
from numba import jit, cuda
from random import randint
import mmh3 as murmurhash3_32  


def hashfunc(m):
    ss = randint(1, 99999999)
    def hash_m(x):
        return murmurhash3_32(x, seed=ss) % m
    return hash_m

class BloomFilter_gpu():
    def __init__(self, n, hash_len):
        self.n = n
        self.hash_len = int(hash_len)
        if (self.n > 0) & (self.hash_len > 0):
            self.k = max(1, int(self.hash_len / n * 0.6931472))
        elif (self.n == 0):
            self.k = 1
        self.h = []
        for i in range(self.k):
            self.h.append(hashfunc(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=np.uint8)

    @cuda.jit
    def insert_kernel(table, k, hashfuncs, keys):
        idx = cuda.grid(1)
        if idx < len(keys):
            key = keys[idx]
            for j in range(k):
                t = hashfuncs[j](key)
                table[t] = 1

    def insert(self, keys):
        if self.hash_len == 0:
            raise SyntaxError('cannot insert to an empty hash table')
        threads_per_block = 256
        blocks_per_grid = (len(keys) + (threads_per_block - 1)) // threads_per_block
        self.insert_kernel[blocks_per_grid, threads_per_block](self.table, self.k, self.h, keys)

    @cuda.jit
    def test_kernel(table, k, hashfuncs, keys, result):
        idx = cuda.grid(1)
        if idx < len(keys):
            match = 0
            for j in range(k):
                t = hashfuncs[j](keys[idx])
                match += 1 * (table[t] == 1)
            if match == k:
                result[idx] = 1

    def test(self, keys, single_key=True):
        result = np.zeros(len(keys), dtype=np.uint8)
        threads_per_block = 256
        blocks_per_grid = (len(keys) + (threads_per_block - 1)) // threads_per_block
        self.test_kernel[blocks_per_grid, threads_per_block](self.table, self.k, self.h, keys, result)
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
