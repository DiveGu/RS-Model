"""
- 运行
- 2021/5/9
"""
import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utils.helper import *
from utils.parser import parse_args
from time import time

from model.BPRMF import BPRMF
from utils.load_data import Data


# 加载预训练的user/item 嵌入
def load_pretrain_data(args):
    pre_model='mf'

    pretrain_path=f'{args.proj_path}pretrain/{args.dataset}/{pre_model}.npz'
    try:
        pretrain_data=np.load(pretrain_path)
        print(f'load the pretrained {pre_model} model params')
    except Exception:
        pretrain_data=None
    return pretrain_data

def main():
    args = parse_args()
    data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
    # 加载数据类 生成batch_data
    data_generator=Data(data_path,args.batch_size)
    data_config=dict()
    data_config['n_users']=data_generator.n_users
    data_config['n_items']=data_generator.n_items

    # 构造pretrain_data
    if args.pretrain in [-1]:
        pretrain_data=load_pretrain_data(args)
    else:
        pretrain_data=None

    # 构造模型
    if(args.model_type=='bprmf'):
        model=BPRMF(data_config,pretrain_data,args)

    # 加载预训练模型参数（tf保存的整个模型参数）
    if args.pretrain==1:
        # TODO
        a=1

    """
    **************************************** 
    训练
    """
    loss_log,pre_log,rec_log,ndcg_log,hit_log=[],[],[],[],[]
    stopping_step=0
    should_stop=False

    # 训练epoch次数
    for epoch in range(args.epoch):
        t1=time()
        loss,mf_loss,reg_loss=0.,0.,0.
        n_batch=data_generator.n_train//args.batch_size+1
        for idx in range(n_batch):
            batch_data=data_generator.generate_train_cf_batch(idx)






if __name__=='__main__':
    main()