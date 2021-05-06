import os
import gc
from loader import load_rate
from splitter import split_test
from helper import *
from utils.parser import parse_args
args = parse_args()


# 生成实验数据
def generate_experiment_data():
    """
    parameters
    dataset : str, dataset name, available options: 'netflix', 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'lastfm', 'bx',
                                                    'amazon-cloth', 'amazon-electronic', 'amazon-book', 'amazon-music',
                                                    'epinions', 'yelp', 'citeulike'
    prepro : str, way to pre-process data, available options: 'origin', '5-core', '10-core'
    test_method : str, way to get test dataset, available options: 'fo', 'loo', 'tloo', 'tfo'

    returns
    """
    experiment_data_path='{}experiment_data/{}/'.format(args.data_path,args.dataset)
    ensureDir(experiment_data_path)
    print(experiment_data_path)
    print(f'start process {args.dataset} with {args.prepro} method...')
    df, uid_2_origin, iid_2_origin = load_rate(args.data_path,args.dataset,args.prepro)
    print(f'test method : {args.test_method}')
    #df[:1000].to_csv(f'./experiment_data/all_{dataset}_{prepro}_{test_method}.csv', index=False)
    train_set, test_set = split_test(df, args.test_method, .2)
    train_set.to_csv(f'{experiment_data_path}/train_{args.dataset}_{args.prepro}_{args.test_method}.csv', index=False)
    test_set.to_csv(f'{experiment_data_path}/test_{args.dataset}_{args.prepro}_{args.test_method}.csv', index=False)
    print('Finish save train and test set...')
    del train_set, test_set, df
    gc.collect()

# 1 预处理数据集、划分数据集
generate_experiment_data()

# 2 为test中用户采样到1000样本（评价指标排序类）

# 3 为train每条数据负采样（pair-wise需要）

from loader import build_candidates_set,get_ur,get_ir

def generate_test_candidates():

    return 0





