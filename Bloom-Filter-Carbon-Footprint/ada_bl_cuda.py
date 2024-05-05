# c CUDA
# %cd /notebooks/Bloom-Filter-Carbon-footprint/Bloom-Filter-Carbon-Footprint
import cupy as cp
import hashlib
from tqdm import tqdm
import math
import pandas as pd
import numpy as np
import hashlib
from random import randint
from sklearn.utils import murmurhash3_32
import time

CUDA_BLOCK_SIZE = 128

class Ada_BloomFilter_gpu:
    def __init__(self, n, hash_len, k_max):
        self.n = n
        self.hash_len = hash_len
        self.table = cp.zeros(self.hash_len, dtype=cp.uintc)
        self.k_max = k_max
        self.seeds = cp.random.randint(1, 99999999, size=k_max, dtype=cp.int32)

        self.insert_kernel = cp.RawKernel(r'''
unsigned int crc32b_m(const unsigned char *str, const unsigned int seed) {
    // Source: https://stackoverflow.com/a/21001712
    unsigned int byte, crc, mask;
    int i = 0, j;
    crc = 0xFFFFFFFF ^ seed;
    while (str[i] != 0) {
        byte = str[i];
        crc = crc ^ byte;
        for (j = 7; j >= 0; j--) {
            mask = -(crc & 1);
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
        i = i + 1;
    }
    return ~crc;
}

extern "C" __global__
void insert_kernel(const unsigned char* url_str, const unsigned int* url_offsets,
        const unsigned int size, const unsigned int* k, const unsigned int hash_len, unsigned int* table) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size) {
      for (unsigned int i = 0; i < k[tid]; i++) {
        unsigned int hash_value = crc32b_m(url_str+url_offsets[tid], i);
        table[hash_value % hash_len] = 1;
      }
    }
}
''', 'insert_kernel')

        self.test_kernel = cp.RawKernel(r'''
unsigned int crc32b_m(const unsigned char *str, const unsigned int seed) {
    // Source: https://stackoverflow.com/a/21001712
    unsigned int byte, crc, mask;
    int i = 0, j;
    crc = 0xFFFFFFFF ^ seed;
    while (str[i] != 0) {
        byte = str[i];
        crc = crc ^ byte;
        for (j = 7; j >= 0; j--) {
            mask = -(crc & 1);
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
        i = i + 1;
    }
    return ~crc;
}

extern "C" __global__
void test_kernel(const unsigned char* url_str, const unsigned int* url_offsets,
        const unsigned int size, const unsigned int* k, const unsigned int hash_len,
        const unsigned int* table, unsigned int* results) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int match = 0;
    if (tid < size) {
      for (unsigned int i = 0; i < k[tid]; i++) {
        unsigned int hash_value = crc32b_m(url_str+url_offsets[tid], i);
        if (table[hash_value % hash_len] > 0) match += 1;
      }
      if (match == k[tid]) results[tid] = 1; 
    }
}
''', 'test_kernel')
            
    def insert(self, url_str_gpu, url_offset_gpu, k_list):
        self.insert_kernel(
            (math.ceil(len(url_offset_gpu)/CUDA_BLOCK_SIZE),),
            (CUDA_BLOCK_SIZE,),
            (url_str_gpu, url_offset_gpu, len(url_offset_gpu), k_list, self.hash_len, self.table )
        )

    def test(self, url_str_gpu, url_offset_gpu, k_list):
        assert(len(url_offset_gpu) == len(k_list))
        # print(url_str_gpu.dtype)
        # print(url_offset_gpu.dtype)
        # print(len(url_offset_gpu))
        # print(k_list.dtype)
        # print(self.table.dtype)
        results = cp.zeros(len(k_list), dtype=cp.uintc)
        self.test_kernel(
            (math.ceil(len(url_offset_gpu)/CUDA_BLOCK_SIZE),),
            (CUDA_BLOCK_SIZE,),
            (url_str_gpu, url_offset_gpu, cp.uintc(len(url_offset_gpu)), k_list, self.hash_len, self.table, results)
        )
        return results
    

def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    timer = Timer()      
    
    c_set = cp.arange(c_min, c_max+10**(-6), 0.1)
    FP_opt = train_negative.shape[0]

    k_min = 0
    
    
    url = positive_sample['url']

    url_str_gpu = cp.frombuffer( ('\0'.join(url) + '\0').encode(), dtype=cp.uint8)
    url_offset_gpu = cp.zeros( len(url,), dtype=cp.uintc)
    # url_length = cp.zeros( (len(url,) )
    url_offset_gpu[0] = 0
    for idx, u in enumerate(url[:-1]):
        # TODO: this part is sequential
        url_offset_gpu[idx+1] = url_offset_gpu[idx] + len(u.encode()) + 1 
                
    for k_max in range(num_group_min, num_group_max + 1):
        for c in c_set:
            timer.start()
            
            tau = cp.sum(c ** cp.arange(0, k_max + 1))
            n = positive_sample.shape[0]
            hash_len = R_sum
            bloom_filter = Ada_BloomFilter_gpu(n, hash_len, k_max)
            thresholds_gpu = cp.zeros(k_max - k_min + 1, dtype=cp.float32)
            thresholds_gpu[-1] = 1.1

            train_negative_score_gpu = cp.asarray(train_negative['score'], dtype=cp.float32)
            num_negative = cp.sum(train_negative_score_gpu <= thresholds_gpu[-1]).item()
            num_piece = int(num_negative / tau) + 1
            score = train_negative.loc[(train_negative_score_gpu.get() <= thresholds_gpu[-1].item()), 'score']
            score = np.sort(score)

            thresholds_cpu = thresholds_gpu.get()
            
            for k in range(k_min, k_max):
                i = k - k_min
                score_1 = score[score < thresholds_cpu[-(i + 1)]]
                if int(num_piece * c ** i) < len(score_1):
                    thresholds_cpu[-(i + 2)] = score_1[-int(num_piece * c ** i)]
                    
            thresholds_gpu = cp.asarray(thresholds_cpu)  # sync gpu cpu data

            url = positive_sample['url']
            score = positive_sample['score']
            score_gpu = cp.asarray(score)
            
            # url_str_gpu = cp.frombuffer( ('\0'.join(url).encode() ), dtype=cp.uint8)
            # url_offset_gpu = cp.zeros( len(url,) )
            # # url_length = cp.zeros( (len(url,) )
            # url_offset_gpu[0] = 0
            # for idx, u in enumerate(url[:-1]):
            #     # TODO: this part is sequential
            #     url_offset_gpu[idx+1] = url_offset_gpu[idx] + len(u.encode()) + 1 
                
            # Bloom filter operations
            # insert part
            a = ( score_gpu.reshape( (-1, 1) ) <
                           cp.broadcast_to(thresholds_gpu, (len(score), *thresholds_gpu.shape))
                )
            b = (a == False) * 0xFFFF + a            
            ix = cp.argmin(b, axis=1)
            k = k_max - ix
            k = cp.asarray(k, dtype=cp.uintc)
            assert(len(k) == len(url_offset_gpu))
            bloom_filter.insert(url_str_gpu,url_offset_gpu, k)
            
            
            # test part
            ML_positive = train_negative.loc[
                (train_negative_score_gpu >= thresholds_cpu[-2]).get()
            , 'url']
            global url_negative
            url_negative = train_negative.loc[
                (train_negative_score_gpu < thresholds_cpu[-2]).get()
            , 'url']
            score_negative = train_negative.loc[
                (train_negative_score_gpu < thresholds_cpu[-2]).get()
            , 'score']
            
            test_result = cp.zeros(len(url_negative))
            
            # prepare negative samples
            url_negative_str_cpu = np.frombuffer( ('\0'.join(url_negative) + '\0').encode(), dtype=np.uint8)
            url_negative_offset_cpu = np.zeros( len(url_negative,), dtype=np.uintc)
            # url_length = cp.zeros( (len(url,) )
            url_negative_offset_cpu[0] = 0
            for idx, u in enumerate(url_negative[:-1]):
                # TODO: this part is sequential
                url_negative_offset_cpu[idx+1] = url_negative_offset_cpu[idx] + len(u.encode()) + 1 
                
            url_negative_str_gpu = cp.asarray(url_negative_str_cpu, dtype=cp.uint8)
            url_negative_offset_gpu = cp.asarray(url_negative_offset_cpu, dtype=cp.uintc)
        
            # test part for loop
            score_negative_gpu = cp.asarray(score_negative, dtype=cp.uintc)
            a = ( score_negative_gpu.reshape( (-1, 1) ) <
                           cp.broadcast_to(thresholds_gpu, (len(score_negative_gpu), *thresholds_gpu.shape))
                )
            b = (a == False) * 0xFFFF + a            
            ix = cp.argmin(b, axis=1)
            k = k_max - ix
            k = cp.asarray(k, dtype=cp.uintc)
            assert(len(k) == len(url_negative_offset_gpu))
            
            test_results_gpu = bloom_filter.test(url_negative_str_gpu, url_negative_offset_gpu, k)
            # FP_items = cp.sum(test_results_gpu).item() + len(ML_positive)
            FP_items = np.sum(test_results_gpu.get()).item() + len(ML_positive)
            
            print(f'False positive items: {FP_items}, Number of groups: {k_max}, c = {c:.2f}')
            if FP_opt > FP_items:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds_gpu.get()  # Copy to host
                k_max_opt = k_max
            
            timer.end()

    return bloom_filter_opt, thresholds_opt, k_max_opt

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        
    def end(self):
        end_time = time.time()
        if self.start_time == None:
            raise RuntimeError("need call start() first")
        print(f'Duration = {end_time-self.start_time:.2}s')
        self.start_time = None
        
def prepare_data():
    data_path = "datasets/URL_data.csv"
    data = pd.read_csv(data_path)
    negative_sample = data.loc[(data['label'] == -1)]
    positive_sample = data.loc[(data['label'] == 1)]
    url_negative = negative_sample['url']
    url = positive_sample['url']
    n = len(url)
    train_negative = negative_sample.sample(frac = 0.3)
    return positive_sample, train_negative

if __name__ == '__main__':
    positive_sample, train_negative = prepare_data()
    
    c_min = 1.8
    c_max = 2.1
    num_group_min = 8
    num_group_max = 12
    R_sum = 200000
    print(Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample))