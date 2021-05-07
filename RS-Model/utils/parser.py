'''
参数设置
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run RS-Model.")
    # 路径参数
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='F:/data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    # 数据集 数据处理参数
    parser.add_argument('--dataset', nargs='?', default='lastfm',
                        help='Choose a dataset from {lastfm,gowalla, yelp2018, amazon-book}')

    parser.add_argument('--prepro',nargs='?',default='20-core',
                        help='Choose data preprocess from {orgin,x-filter,x-core}')

    parser.add_argument('--test_method',nargs='?',default='ufo',
                        help='Choose a way to get test dataset from {fo, loo, tloo, tfo}')

    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
  
    return parser.parse_args()

