"""
加载处理好的数据
构造u-i邻接矩阵,生成用历史行为表示u或i的数据形式
生成batch样本
"""
import json
from time import time
import numpy as np
import pandas as pd
import random as rd
import scipy.sparse as sp
from utils.load_data import Data

class Data_History(Data):
    def __init__(self, path, batch_size):
        super(Data_History,self).__init__(path, batch_size)

    # 获取u2i交互矩阵
    def get_r_matrix(self):
        try:
            t0 = time()
            r_matrix = sp.load_npz(self.path + '/s_r_matrix.npz')
            print('already load rate matrix {} {:.1}s'.format(r_matrix.shape,time()-t0))

        except Exception:
            r_matrix = self._creat_r_matrix()
            sp.save_npz(self.path + '/s_r_matrix.npz', r_matrix)

        return r_matrix

    def _creat_r_matrix(self):
        # 通过train数据构造 ui 评分矩阵R
        R=sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for uid,items in self.train_user_dict.items():
            for iid in items:
                R[uid,iid]=1.

        return R.tocsr()

    # 获取一个batch里的用户的行为序列+pad表示
    def _get_batch_user_history(self,uid_list):
        user_his=[]
        cur_batch_his_len=[]
        for uid in uid_list:
            user_his.append(self.train_user_dict[uid])
            cur_batch_his_len.append(len(user_his[-1]))

        user_his=self._add_mask(self.n_items,user_his,max(cur_batch_his_len))
        return np.array(user_his)

    # 使用mask_num来填充每个his_list的元素 填充到max_len
    def _add_mask(self, mask_num, his_list, max_len):
        for i in range(len(his_list)):
            his_list[i] = his_list[i] + [mask_num]*(max_len-len(his_list[i]))
        return his_list

    # 生成训练batch
    def generate_train_cf_batch(self,idx):
        # 1 父类batch_data:[user,pos_item,neg_item]
        batch_data=super(Data_History,self).generate_train_cf_batch(idx)
        # 2 新增user_history
        batch_data['user_his_item']=self._get_batch_user_history(list(batch_data['user'].reshape(-1,)))

        return batch_data 

    # 生成batch的feed_dict字典
    def generate_train_feed_dict(self,model,batch_data):

        feed_dict={
            model.users:batch_data['user'],
            model.pos_items:batch_data['pos_item'],
            model.neg_items:batch_data['neg_item'],
            model.user_his_item:batch_data['user_his_item'],
        }
        
        return feed_dict    



#np.set_printoptions(threshold=10000) 
#from utils.parser import parse_args
#args = parse_args()
#data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
## 加载数据类 生成batch_data
#data_generator=Data_History(data_path,args.batch_size)
#print(data_generator.generate_train_cf_batch(0))
