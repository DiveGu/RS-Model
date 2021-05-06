import os
import gc
from loader import load_rate
from splitter import split_test


# 生成实验数据
def generate_experiment_data(dataset, prepro, test_method):
    """
    parameters
    dataset : str, dataset name, available options: 'netflix', 'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m', 'lastfm', 'bx',
                                                    'amazon-cloth', 'amazon-electronic', 'amazon-book', 'amazon-music',
                                                    'epinions', 'yelp', 'citeulike'
    prepro : str, way to pre-process data, available options: 'origin', '5-core', '10-core'
    test_method : str, way to get test dataset, available options: 'fo', 'loo', 'tloo', 'tfo'

    returns
    """
    if not os.path.exists('./experiment_data/'):
        os.makedirs('./experiment_data/')
    print(f'start process {dataset} with {prepro} method...')
    df, user_num, item_num = load_rate(dataset, prepro, True)
    print(f'test method : {test_method}')
    #df[:1000].to_csv(f'./experiment_data/all_{dataset}_{prepro}_{test_method}.csv', index=False)
    train_set, test_set = split_test(df, test_method, .2)
    train_set.to_csv(f'./experiment_data/train_{dataset}_{prepro}_{test_method}.csv', index=False)
    test_set.to_csv(f'./experiment_data/test_{dataset}_{prepro}_{test_method}.csv', index=False)
    print('Finish save train and test set...')
    del train_set, test_set, df
    gc.collect()

generate_experiment_data('lastfm', '5-filter', 'ufo')

# 1 预处理数据集、划分数据集

# 2 为test中用户采样到1000样本（评价指标排序类）

# 3 为train每条数据负采样（pair-wise需要）

from loader import build_candidates_set,get_ur,get_ir

def generate_test_candidates():

    return 0

