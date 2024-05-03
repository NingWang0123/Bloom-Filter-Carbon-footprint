import pandas as pd
import math
import numpy as np
import operator
from random import shuffle
from bloom_filter_self_def import BloomFilter
import random
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import kruskal
from itertools import product
from gpu_py import simus,parallel_simus
import os

num_cpu=os.cpu_count()

np.random.seed(123)
random.seed(123)

items_counts = 20
fp_prob = 0.1

mu, sigma = 0.7, 0.2
scores_pre = np.random.normal(mu, sigma, 1000)
scores_pre = [item for item in scores_pre if (item >= 0 and item<=1.0)]

mu, sigma = 0.3, 0.2
scores_upre = np.random.normal(mu, sigma, 1000)
scores_upre=[item for item in scores_upre if (item >= 0 and item<=1.0)]

query_items=scores_pre+scores_upre
shuffle(query_items)

scores_pre_np=np.array(scores_pre)
query_items_np=np.array(query_items)

scores_pre_pd=pd.Series(scores_pre)
query_items_pd=pd.Series(query_items)

scores_pre_df=pd.DataFrame(scores_pre).iloc[:,0]
query_items_df=pd.DataFrame(query_items).iloc[:,0]

def bf_simu(scores_pre_series, query_items_series, items_counts, fp_prob):
    bf = BloomFilter(items_counts, fp_prob)

    for item in scores_pre_series:
        bf.add(str(item))

    i = 0
    j = 0

    for item in query_items_series:
        if bf.check(str(item)):
            if item in scores_pre_series.values:
                i += 1
            else:
                j += 1

    return i, j



def bf_simu2(scores_pre_series, query_items_series, items_counts, fp_prob):
    bf = BloomFilter(items_counts, fp_prob)

    for item in scores_pre_series:
        bf.add(str(item))

    i = 0
    j = 0

    for item in query_items_series:
        if bf.check(str(item)):
            if item in scores_pre_series:
                i += 1
            else:
                j += 1

    return i, j

n=35

np_args=(scores_pre_np, query_items_np, items_counts, fp_prob)
lst_args=(scores_pre, query_items, items_counts, fp_prob)
pd_args=(scores_pre_pd, query_items_pd, items_counts, fp_prob)
df_args=(scores_pre_df, query_items_df, items_counts, fp_prob)


df_np=simus(bf_simu2,*np_args,num_runs=n)
df_lst=simus(bf_simu2,*lst_args,num_runs=n)
df_pd=simus(bf_simu,*pd_args,num_runs=n)
df_df=simus(bf_simu,*df_args,num_runs=n)


df_np_para=parallel_simus(bf_simu2,*np_args,num_runs=n,num_cpus=num_cpu)
df_lst_para=parallel_simus(bf_simu2,*lst_args,num_runs=n,num_cpus=num_cpu)
df_pd_para=parallel_simus(bf_simu,*pd_args,num_runs=n,num_cpus=num_cpu)
df_df_para=parallel_simus(bf_simu,*df_args,num_runs=n,num_cpus=num_cpu)

df_np['type']='np'
df_lst['type']='list'
df_pd['type']='pd'
df_df['type']='df'


df_np_para['type']='np'
df_lst_para['type']='list'
df_pd_para['type']='pd'
df_df_para['type']='df'

df_np['running_method']='seq'
df_lst['running_method']='seq'
df_pd['running_method']='seq'
df_df['running_method']='seq'


df_np_para['running_method']='parallel'
df_lst_para['running_method']='parallel'
df_pd_para['running_method']='parallel'
df_df_para['running_method']='parallel'

df_all=pd.concat([df_lst, df_np,df_pd,df_df,
                 df_np_para,df_lst_para,df_pd_para,df_df_para], ignore_index=True)
df_all.to_csv('D:\\Desktop\\repos\\Bloom-Filter-Carbon-Footprint\\emissions_dif_dt_bl.csv', index=False)