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


#######
###cupy 

import concurrent.futures
import cupy as cp

def Find_Optimal_Parameters_CONCUR(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = cp.arange(c_min, c_max + 10 ** (-6), 0.1)
    FP_opt = train_negative.shape[0]

    for num_group in range(num_group_min, num_group_max + 1):
        for c in c_set:
            thresholds = cp.zeros(num_group + 1, dtype=cp.float32)
            thresholds[0] = -0.1
            thresholds[-1] = 1.1
            num_negative = train_negative.shape[0]
            tau = cp.sum(c ** cp.arange(0, num_group, 1))
            num_piece = int(num_negative / tau)
            score = cp.sort(cp.array(list(train_negative['score'])))

            for i in range(1, num_group):
                if thresholds[-i] > 0:
                    score_1 = score[score < thresholds[-i]]
                    if int(num_piece * c ** (i - 1)) <= len(score_1):
                        thresholds[-(i + 1)] = score_1[-int(num_piece * c ** (i - 1))]
                    else:
                        thresholds[-(i + 1)] = 0
                else:
                    thresholds[-(i + 1)] = 1

            count_nonkey = cp.zeros(num_group)
            for j in range(num_group):
                count_nonkey[j] = cp.sum((score >= thresholds[j]) & (score < thresholds[j + 1]))

            num_group_1 = int(cp.sum(count_nonkey > 0))
            count_nonkey = count_nonkey[count_nonkey > 0]
            thresholds = thresholds[-(num_group_1 + 1):]

            count_key = cp.zeros(num_group_1)
            url_group = []
            bloom_filter = []
            positive_scores = cp.asarray(positive_sample['score'])
            for j in range(num_group_1):
                mask = (positive_scores >= thresholds[j]) & (positive_scores < thresholds[j + 1])
                count_key[j] = cp.sum(mask)
                urls_in_group = positive_sample['url'][cp.asnumpy(mask)].tolist()  # Convert mask back to NumPy for indexing Pandas DataFrame
                url_group.append(urls_in_group)

            R = cp.zeros(num_group_1 - 1)
            R[:] = 0.5 * R_sum
            non_empty_ix = min(cp.where(count_key > 0)[0])
            if non_empty_ix > 0:
                R[0:non_empty_ix] = 0
            kk = 1
            while cp.abs(cp.sum(R) - R_sum) > 200:
                if (cp.sum(R) > R_sum):
                    R[non_empty_ix] = R[non_empty_ix] - int((0.5 * R_sum) * (0.5) ** kk + 1)
                else:
                    R[non_empty_ix] = R[non_empty_ix] + int((0.5 * R_sum) * (0.5) ** kk + 1)
                R[non_empty_ix:] = R_size(count_key[non_empty_ix:-1], count_nonkey[non_empty_ix:-1], R[non_empty_ix])
                if int((0.5 * R_sum) * (0.5) ** kk + 1) == 1:
                    break
                kk += 1

            Bloom_Filters = []
            for j in range(int(num_group_1 - 1)):
                if j < non_empty_ix:
                    Bloom_Filters.append([0])
                else:
                    Bloom_Filters.append(BloomFilter_gpu(count_key[j], int(R[j])))
                    Bloom_Filters[j].insert(url_group[j])
                    # print('insert_1')

            threshold_value = float(cp.asnumpy(thresholds[-2]))

            ML_positive = train_negative.loc[(train_negative['score'] >= threshold_value), 'url']
            url_negative = train_negative.loc[(train_negative['score'] < threshold_value), 'url']

            score_negative = train_negative.loc[(train_negative['score'] < threshold_value), 'score']

            test_result = cp.zeros(len(url_negative))
            def process_data(ss, score_s, url_s):
                indices = cp.where(score_s < thresholds)[0]
                if indices.size > 0:
                    ix = int(indices.min()) - 1
                    if ix >= 0 and ix < len(Bloom_Filters):
                        if ix >= non_empty_ix:
                            result = Bloom_Filters[ix].test(url_s)
                        else:
                            result = 0
                    else:
                        print("Index out of range. Adjusted index:", ix, "Length of Bloom_Filters:", len(Bloom_Filters))
                        result = 0
                else:
                    print("No valid index found.")
                    result = 0
                return ss, result

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_data, ss, score_s, url_s) for ss, (score_s, url_s) in enumerate(zip(score_negative, url_negative))]
                for future in concurrent.futures.as_completed(futures):
                    ss, result = future.result()
                    test_result[ss] = result


            # ss = 0
            # for score_s, url_s in zip(score_negative, url_negative):
            #     # if ss == 50:
            #     #     break
            #     indices = cp.where(score_s < thresholds)[0]
            #     if indices.size > 0:
            #         ix = int(indices.min()) - 1
            #         if ix >= 0 and ix < len(Bloom_Filters):
            #             if ix >= non_empty_ix:
            #                 test_result[ss] = Bloom_Filters[ix].test(url_s)
            #                 # print('test_1')
            #             else:
            #                 test_result[ss] = 0
            #         else:
            #             print("Index out of range. Adjusted index:", ix, "Length of Bloom_Filters:", len(Bloom_Filters))
            #             test_result[ss] = 0
            #     else:
            #         print("No valid index found.")
                # ss += 1
            FP_items = cp.sum(test_result) + len(ML_positive)
            print(f'False positive items: {FP_items}, Number of groups: {num_group}, c = {round(float(c), 2)}')
            if FP_opt > FP_items:
                FP_opt = FP_items
                Bloom_Filters_opt = Bloom_Filters
                thresholds_opt = thresholds
                non_empty_ix_opt = non_empty_ix

    return Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt