import torch
import numpy as np

def hashfunc(length):
    # Assuming a simple hash function for demonstration. Replace with an appropriate hash function.
    return lambda x: hash(x) % length

class Ada_BloomFilter():
    def __init__(self, n, hash_len, k_max):
        self.n = n
        self.hash_len = int(hash_len)
        self.h = []
        for i in range(int(k_max)):
            self.h.append(hashfunc(self.hash_len))
        # Initialize the table on GPU
        self.table = torch.zeros(self.hash_len, dtype=torch.int32, device='cuda')
    
    def insert(self, key, k):
        for j in range(int(k)):
            t = self.h[j](key)
            self.table[t] = 1
    
    def test(self, key, k):
        match = 0
        for j in range(int(k)):
            t = self.h[j](key)
            match += 1 * (self.table[t] == 1)
        return int(match == k)

def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = torch.arange(c_min, c_max + 10**(-6), 0.1, device='cuda')
    FP_opt = train_negative.shape[0]

    k_min = 0
    for k_max in range(num_group_min, num_group_max + 1):
        for c in c_set:
            tau = torch.sum(c ** torch.arange(0, k_max - k_min + 1, 1, device='cuda'))
            n = positive_sample.shape[0]
            hash_len = R_sum
            bloom_filter = Ada_BloomFilter(n, hash_len, k_max)
            thresholds = torch.ones(k_max - k_min + 1, device='cuda') * 1.1
            # Convert the last threshold to a numpy float to use in comparison with Pandas Series
            last_threshold = thresholds[-1].item()
            num_negative = (train_negative['score'] <= last_threshold).sum()
            num_piece = int(num_negative / tau.item()) + 1
            score = train_negative.loc[train_negative['score'] <= last_threshold, 'score'].values
            score = torch.tensor(np.sort(score), device='cuda')
            for k in range(k_min, k_max):
                i = k - k_min
                score_1 = score[score < torch.tensor(thresholds[-(i + 1)].item(), device='cuda')]
                if int(num_piece * c.item() ** i) < len(score_1):
                    thresholds[-(i + 2)] = score_1[-int(num_piece * c.item() ** i)]

            url = positive_sample['url']
            score = positive_sample['score'].values
            for score_s, url_s in zip(score, url):
                ix = min((torch.tensor(score_s, device='cuda') < thresholds).nonzero().cpu().numpy()[0])
                k = k_max - ix
                bloom_filter.insert(url_s, k)

            url_negative = train_negative.loc[train_negative['score'] < thresholds[-2].item(), 'url']
            score_negative = train_negative.loc[train_negative['score'] < thresholds[-2].item(), 'score'].values
            test_result = torch.zeros(len(url_negative), device='cuda')
            ss = 0
            for score_s, url_s in zip(score_negative, url_negative):
                ix = min((torch.tensor(score_s, device='cuda') < thresholds).nonzero().cpu().numpy()[0])
                k = k_max - ix
                test_result[ss] = bloom_filter.test(url_s, k)
                ss += 1
            FP_items = torch.sum(test_result).item() + len(url_negative)
            print(f'False positive items: {FP_items}, Number of groups: {k_max}, c = {round(c.item(), 2)}')

            if FP_opt > FP_items:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds
                k_max_opt = k_max

    return bloom_filter_opt, thresholds_opt, k_max_opt

############
#### 2nd version


def hashfunc(m):
    ss = randint(1, 99999999)
    def hash_m(x):
        # Assuming x is a string, convert to bytes for hashing
        return murmurhash.mrmr.hash(x.encode('utf-8'), seed=ss) % m
    return hash_m

class Ada_BloomFilter():
    def __init__(self, n, hash_len, k_max):
        self.n = n
        self.hash_len = int(hash_len)
        self.h = [hashfunc(self.hash_len) for _ in range(k_max)]
        # Initialize the table on GPU
        self.table = torch.zeros(self.hash_len, dtype=torch.int32, device='cuda')
    
    def insert(self, keys):
        positions = torch.zeros((len(keys), len(self.h)), dtype=torch.long, device='cuda')
        for i, key in enumerate(keys):
            for j, hash_func in enumerate(self.h):
                positions[i, j] = hash_func(key)
        # Flatten and remove duplicates before updating the table to prevent race conditions
        positions = positions.view(-1).unique()
        self.table[positions] = 1
    
    def test(self, keys):
        results = torch.ones(len(keys), dtype=torch.int32, device='cuda')
        for i, key in enumerate(keys):
            for j, hash_func in enumerate(self.h):
                if not self.table[hash_func(key)]:
                    results[i] = 0
                    break
        return results

def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = torch.arange(c_min, c_max + 10**(-6), 0.1, device='cuda')
    FP_opt = train_negative.shape[0]

    for k_max in range(num_group_min, num_group_max + 1):
        for c in c_set:
            tau = torch.sum(c ** torch.arange(0, k_max + 1, device='cuda'))
            n = positive_sample.shape[0]
            hash_len = R_sum
            bloom_filter = Ada_BloomFilter(n, hash_len, k_max)
            thresholds = torch.zeros(k_max + 1, device='cuda')
            thresholds[-1] = 1.1  # Define the initial threshold
            num_negative = (train_negative['score'] <= thresholds[-1].item()).sum()
            num_piece = int(num_negative / tau.item()) + 1
            score = train_negative[train_negative['score'] <= thresholds[-1].item()]['score'].values
            score = torch.tensor(np.sort(score), device='cuda')
            for k in range(k_max):
                i = k
                score_1 = score[score < thresholds[-(i + 1)].item()]
                if int(num_piece * c.item() ** i) < len(score_1):
                    thresholds[-(i + 2)] = score_1[-int(num_piece * c.item() ** i)]

            keys = [url_s for score_s, url_s in zip(positive_sample['score'].values, positive_sample['url']) if score_s < thresholds[0].item()]
            bloom_filter.insert(keys)

            test_keys = [url_s for score_s, url_s in zip(train_negative['score'].values, train_negative['url']) if score_s < thresholds[0].item()]
            test_results = bloom_filter.test(test_keys)
            FP_items = test_results.sum().item()
            print(f'False positive items: {FP_items}, Number of groups: {k_max}, c = {round(c.item(), 2)}')

            if FP_opt > FP_items:
                FP_opt = FP_items
                bloom_filter_opt = bloom_filter
                thresholds_opt = thresholds
                k_max_opt = k_max

    return bloom_filter_opt, thresholds_opt, k_max_opt