import pandas as pd
import math
import numpy as np
import time
from codecarbon import EmissionsTracker
from Bloom_filter import BloomFilter
import PLBF
import disjoint_Ada_BF
import learned_Bloom_filter
import Ada_BF
from codecarbon import EmissionsTracker
from concurrent.futures import ThreadPoolExecutor
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
num_cpu=os.cpu_count()
data_path=data_path = "D:\\Desktop\\repos\\Bloom-Filter-Carbon-Footprint\\datasets\\URL_data.csv"
data = pd.read_csv(data_path)
negative_sample = data.loc[(data['label'] == -1)]
positive_sample = data.loc[(data['label'] == 1)]
url_negative = negative_sample['url']
url = positive_sample['url']
n = len(url)
train_negative = negative_sample.sample(frac = 0.3)


def get_train_time(num_group_min, num_group_max, c_min, c_max, R_sum,url,n,min_thres,max_thres,train_negative,positive_sample,model):
    if model=='bl':
        bloom_filter = BloomFilter(n, R_sum)
        bloom_filter.insert(url)
        return bloom_filter
    elif model=='disjoint_Ada_BF':
        Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt = disjoint_Ada_BF.Find_Optimal_Parameters(c_min,c_max, num_group_min, 
                                                                   num_group_max, R_sum,
                                                                   train_negative, positive_sample)
        return Bloom_Filters_opt,thresholds_opt,non_empty_ix_opt
    elif model=='learned_BF':
        bloom_filter_opt, thres_opt = learned_Bloom_filter.Find_Optimal_Parameters(max_thres, min_thres, R_sum,
                                                                                     train_negative, positive_sample)
        return bloom_filter_opt,thres_opt
            
    
    elif model=='Ada_BF':
        bloom_filter_opt, thresholds_opt, k_max_opt = Ada_BF.Find_Optimal_Parameters(c_min, c_max, num_group_min,
                                                                                      num_group_max, 
                                                                                      R_sum, train_negative,
                                                                                      positive_sample)
        return bloom_filter_opt, thresholds_opt, k_max_opt
    else:
        return('invalid model, model must be one of bl,disjoint_Ada_BF,learned_BF,Ada_BF')



def simulation_function(func, *args):
    tracker = EmissionsTracker()
    tracker.start()
    start_time = time.time()
    result = func(*args)  
    end_time = time.time()
    time_elapsed = end_time - start_time
    emissions: float = tracker.stop()
    return time_elapsed, emissions


def simus(func, *args, num_runs=100):
    times = np.zeros(num_runs)
    electricity = np.zeros(num_runs)

    for i in range(num_runs):
        times[i], electricity[i] = simulation_function(func, *args) 

    return pd.DataFrame({'total_time': times, 'electricity': electricity})

def parallel_simus(func, *args, num_runs=100, num_cpus=None):
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(lambda _: simulation_function(func, *args), range(num_runs)))

    times, electricity = zip(*results)
    times = np.array(times)
    electricity = np.array(electricity)

    return pd.DataFrame({'total_time': times, 'electricity': electricity})

n_iters=35

args_bl=(8, 12, 1.8, 2.1, 200000,url,n,0.5,0.9,
                           train_negative,positive_sample,'bl')
args_dis_ada=(8, 12, 1.8, 2.1, 200000,url,n,0.5,0.9,
                           train_negative,positive_sample,'disjoint_Ada_BF')
args_lbf=(8, 12, 1.8, 2.1, 200000,url,n,0.5,0.9,
                           train_negative,positive_sample,'learned_BF')
args_ada=(8, 12, 1.8, 2.1, 200000,url,n,0.5,0.9,
                           train_negative,positive_sample,'Ada_BF')
                           
df_bl=simus(get_train_time,*args_bl,num_runs=n_iters)

df_dis_ada=simus(get_train_time,*args_dis_ada,num_runs=n_iters)

df_lbf=simus(get_train_time,*args_lbf,num_runs=n_iters)

df_ada=simus(get_train_time,*args_ada,num_runs=n_iters)

df_bl_para=parallel_simus(get_train_time,*args_bl,num_runs=n_iters,num_cpus=num_cpu)

df_dis_ada_para=parallel_simus(get_train_time,*args_dis_ada,num_runs=n_iters,num_cpus=num_cpu)

df_lbf_para=parallel_simus(get_train_time,*args_lbf,num_runs=n_iters,num_cpus=num_cpu)

df_ada_para=parallel_simus(get_train_time,*args_ada,num_runs=n_iters,num_cpus=num_cpu)

df_bl['method']='bloom_filter'
df_dis_ada['method']='disjoint_Ada_BF'
df_lbf['method']='learned_bf'
df_ada['method']='Ada_BF'

df_bl_para['method']='bloom_filter'
df_dis_ada_para['method']='disjoint_Ada_BF'
df_lbf_para['method']='learned_bf'
df_ada_para['method']='Ada_BF'

df_bl['running_method']='seq'
df_dis_ada['running_method']='seq'
df_lbf['running_method']='seq'
df_ada['running_method']='seq'

df_bl_para['running_method']='parallel'
df_dis_ada_para['running_method']='parallel'
df_lbf_para['running_method']='parallel'
df_ada_para['running_method']='parallel'

df_bl['stage']='training'
df_dis_ada['stage']='training'
df_lbf['stage']='training'
df_ada['stage']='training'

df_bl_para['stage']='training'
df_dis_ada_para['stage']='training'
df_lbf_para['stage']='training'
df_ada_para['stage']='training'

df_all_train=pd.concat([df_bl, df_dis_ada,df_lbf,df_ada,
                        df_bl_para,df_dis_ada_para,df_lbf_para,df_ada_para], ignore_index=True)


bloom_filter=get_train_time(8, 12, 1.8, 2.1, 200000,url,n,0.5,0.9,
                           train_negative,positive_sample,model='bl')

Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt=get_train_time(8, 12, 1.8, 2.1, 200000,url,n,0.5,0.9,
                           train_negative,positive_sample,model='disjoint_Ada_BF')

bloom_filter_opt,thres_opt=get_train_time(8, 12, 1.8, 2.1, 200000,url,n,0.5,0.9,
                           train_negative,positive_sample,model='learned_BF')

bloom_filter_opt_ada, thresholds_opt_ada, k_max_opt=get_train_time(8, 12, 1.8, 2.1, 200000,url,n,0.5,0.9,
                           train_negative,positive_sample,model='Ada_BF')\



def get_testing_time(data,bloom_filter,Bloom_Filters_opt,
                     thresholds_opt,non_empty_ix_opt,bloom_filter_opt,
                    thres_opt,bloom_filter_opt_ada,thresholds_opt_ada,k_max_opt,model):
    
    if model=='bl':
        negative_sample = data.loc[(data['label']==-1)]
        url_negative = negative_sample['url']
        n1 = bloom_filter.test(url_negative, single_key=False)
        print('False positive items: ', sum(n1))
        
    elif model=='disjoint_Ada_BF':
        negative_sample = data.loc[(data['label']==-1)]
        ML_positive = negative_sample.loc[(negative_sample['score'] >= thresholds_opt[-2]), 'url']
        url_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt[-2]), 'url']
        score_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt[-2]), 'score']
        test_result = np.zeros(len(url_negative))
        ss = 0
        for score_s, url_s in zip(score_negative, url_negative):
            ix = min(np.where(score_s < thresholds_opt)[0]) - 1
            if ix >= non_empty_ix_opt:
                test_result[ss] = Bloom_Filters_opt[ix].test(url_s)
            else:
                test_result[ss] = 0
            ss += 1
        FP_items = sum(test_result) + len(ML_positive)
        FPR = FP_items/len(url_negative)
        print('False positive items: {}; FPR: {}; Size of quries: {}'.format(FP_items, FPR, len(url_negative)))
        
    elif model=='learned_BF':
        negative_sample = data.loc[(data['label']==-1)]
        ML_positive = negative_sample.loc[(negative_sample['score'] > thres_opt), 'url']
        bloom_negative = negative_sample.loc[(negative_sample['score'] <= thres_opt), 'url']
        score_negative = negative_sample.loc[(negative_sample['score'] < thres_opt), 'score']
        BF_positive = bloom_filter_opt.test(bloom_negative, single_key = False)
        FP_items = sum(BF_positive) + len(ML_positive)
        print('False positive items: %d' % FP_items)

    elif model=='Ada_BF':
        negative_sample = data.loc[(data['label']==-1)]
        ML_positive = negative_sample.loc[(negative_sample['score'] >= thresholds_opt_ada[-2]), 'url']
        url_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt_ada[-2]), 'url']
        score_negative = negative_sample.loc[(negative_sample['score'] < thresholds_opt_ada[-2]), 'score']
        test_result = np.zeros(len(url_negative))
        ss = 0
        for score_s, url_s in zip(score_negative, url_negative):
            ix = min(np.where(score_s < thresholds_opt_ada)[0])
            # thres = thresholds[ix]
            k = k_max_opt - ix
            test_result[ss] = bloom_filter_opt_ada.test(url_s, k)
            ss += 1
        FP_items = sum(test_result) + len(ML_positive)
        print('False positive items: %d' % FP_items)
       
    else:
        return('invalid model, model must be one of bl,disjoint_Ada_BF,learned_BF,Ada_BF')
    
func_args_bl = (data, bloom_filter, Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt,
                bloom_filter_opt, thres_opt, bloom_filter_opt_ada, thresholds_opt_ada,
                k_max_opt, 'bl')
func_args_dis_ada= (data, bloom_filter, Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt,
                bloom_filter_opt, thres_opt, bloom_filter_opt_ada, thresholds_opt_ada,
                k_max_opt, 'disjoint_Ada_BF')
func_args_lbf= (data, bloom_filter, Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt,
                bloom_filter_opt, thres_opt, bloom_filter_opt_ada, thresholds_opt_ada,
                k_max_opt, 'learned_BF')
func_args_ada = (data, bloom_filter, Bloom_Filters_opt, thresholds_opt, non_empty_ix_opt,
                bloom_filter_opt, thres_opt, bloom_filter_opt_ada, thresholds_opt_ada,
                k_max_opt, 'Ada_BF')

df_bl_test=simus(get_testing_time, *func_args_bl, num_runs=n_iters)

df_dis_ada_test=simus(get_testing_time,*func_args_dis_ada,num_runs=n_iters)

df_lbf_test=simus(get_testing_time,*func_args_lbf,num_runs=n_iters)

df_ada_test=simus(get_testing_time,*func_args_ada,num_runs=n_iters)



df_bl_test_para=parallel_simus(get_testing_time,*func_args_bl,num_runs=n_iters,num_cpus=num_cpu)

df_dis_ada_test_para=parallel_simus(get_testing_time,*func_args_bl,num_runs=n_iters,num_cpus=num_cpu)

df_lbf_test_para=parallel_simus(get_testing_time,*func_args_dis_ada,num_runs=n_iters,num_cpus=num_cpu)

df_ada_test_para=parallel_simus(get_testing_time,*func_args_ada,num_runs=n_iters,num_cpus=num_cpu)

df_bl_test['method']='bloom_filter'
df_dis_ada_test['method']='disjoint_Ada_BF'
df_lbf_test['method']='learned_bf'
df_ada_test['method']='Ada_BF'

df_bl_test_para['method']='bloom_filter'
df_dis_ada_test_para['method']='disjoint_Ada_BF'
df_lbf_test_para['method']='learned_bf'
df_ada_test_para['method']='Ada_BF'

df_bl_test['running_method']='seq'
df_dis_ada_test['running_method']='seq'
df_lbf_test['running_method']='seq'
df_ada_test['running_method']='seq'
x
df_bl_test_para['running_method']='parallel'
df_dis_ada_test_para['running_method']='parallel'
df_lbf_test_para['running_method']='parallel'
df_ada_test_para['running_method']='parallel'

df_bl_test['stage']='test'
df_dis_ada_test['stage']='test'
df_lbf_test['stage']='test'
df_ada_test['stage']='test'

df_bl_test_para['stage']='test'
df_dis_ada_test_para['stage']='test'
df_lbf_test_para['stage']='test'
df_ada_test_para['stage']='test'

df_all_test=pd.concat([df_bl_test, df_dis_ada_test,df_lbf_test,df_ada_test,
                      df_bl_test_para,df_dis_ada_test_para,df_lbf_test_para,df_ada_test_para], ignore_index=True)

df_all=pd.concat([df_all_train,df_all_test], ignore_index=True)

df_all['running']='cpu'
df_all.to_csv('D:\\Desktop\\repos\\Bloom-Filter-Carbon-Footprint\\emissions_bl_cpu_test.csv', index=False)