def hashfunc(length):
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._insert_single, key, self.h[j]) for j in range(int(k))]
            for future in concurrent.futures.as_completed(futures):
                pass

    def _insert_single(self, key, hash_func):
        t = hash_func(key)
        self.table[t] = 1
    
    def test(self, key, k):
        match = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._test_single, key, self.h[j]) for j in range(int(k))]
            for future in concurrent.futures.as_completed(futures):
                match += future.result()
        return int(match == k)

    def _test_single(self, key, hash_func):
        t = hash_func(key)
        return int(self.table[t] == 1)
    
    
def Find_Optimal_Parameters(c_min, c_max, num_group_min, num_group_max, R_sum, train_negative, positive_sample):
    c_set = torch.arange(c_min, c_max + 10**(-6), 0.1, device='cuda')
    FP_opt = train_negative.shape[0]
    k_min = 0
    optimal_params = []

    def evaluate_params(c, k_max):
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
        return FP_items, k_max, c.item()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluate_params, c, k_max) for c in c_set for k_max in range(num_group_min, num_group_max + 1)]
        for future in concurrent.futures.as_completed(futures):
            FP_items, k_max, c_value = future.result()
            print(f'False positive items: {FP_items}, Number of groups: {k_max}, c = {round(c_value, 2)}')
            if FP_opt > FP_items:
                FP_opt = FP_items
                optimal_params = [FP_opt, k_max, c_value]

    return optimal_params
