import os
import gc
import json
from loader_origin import load_rate,build_candidates_set,get_ur,get_ir
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
    experiment_data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
    ensureDir(experiment_data_path)
    print(experiment_data_path)
    print(f'start process {args.dataset} with {args.prepro} method...')
    # 1 预处理数据集、划分数据集
    df, uid_2_origin, iid_2_origin = load_rate(args.data_path,args.dataset,args.prepro)
    print(f'test method : {args.test_method}')
    # train-test划分
    train_set, test_set = split_test(df, args.test_method, .2)
    # 保存实验数据
    f = open(f'{experiment_data_path}uid_dict.txt','w')
    f.write(str(uid_2_origin))
    f.close()
    f = open(f'{experiment_data_path}iid_dict.txt','w')
    f.write(str(iid_2_origin))
    f.close()
    train_set.to_csv(f'{experiment_data_path}train.csv',index=False)
    test_set.to_csv(f'{experiment_data_path}test.csv',index=False)
    #train_set.to_csv(f'{experiment_data_path}/train_{args.dataset}_{args.prepro}_{args.test_method}.csv', index=False)
    #test_set.to_csv(f'{experiment_data_path}/test_{args.dataset}_{args.prepro}_{args.test_method}.csv', index=False)
    
    # 2 为test中用户采样到1000样本（评价指标排序类）
    train_ur=get_ur(train_set)
    test_ur=get_ur(test_set)
    test_candidate_neg=build_candidates_set(test_ur, train_ur, set(range(len(iid_2_origin))), candidates_num=100)
    # 保存test的负采样
    f = open(f'{experiment_data_path}test_neg.txt','w')
    f.write(json.dumps(test_candidate_neg))
    #f.write(str(test_candidate_neg))
    f.close()

    print('Finish save train and test set...')
    del train_set, test_set, df
    gc.collect()
    # 3 为train每条数据负采样（pair-wise需要） fix or random


generate_experiment_data()

