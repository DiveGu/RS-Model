"""
加载处理好的数据
构造u-i邻接矩阵
生成batch样本
"""
import json
from time import time
import numpy as np
import pandas as pd
import random as rd
import scipy.sparse as sp
from utils.load_data import Data

class Data_Graph(Data):
    def __init__(self, path, batch_size):
        super(Data_Graph,self).__init__(path, batch_size)
        # 通过train数据构造 ui 评分矩阵R
        self.R=sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for uid,items in self.train_user_dict.items():
            for iid in items:
                self.R[uid,iid]=1.

    def get_adj_matrix(self):
        try:
            t0 = time()
            adj_matrix = sp.load_npz(self.path + '/s_adj_matrix.npz')
            norm_adj_matrix = sp.load_npz(self.path + '/s_norm_adj_matrix.npz')
            mean_adj_matrix = sp.load_npz(self.path + '/s_mean_adj_matrix.npz')
            print('already load adj matrix {} {:.1}s'.format(adj_matrix.shape, time() - t0))

        except Exception:
            adj_matrix, norm_adj_matrix, mean_adj_matrix = self.creat_adj_matrix()
            sp.save_npz(self.path + '/s_adj_matrix.npz', adj_matrix)
            sp.save_npz(self.path + '/s_norm_adj_matrix.npz', norm_adj_matrix)
            sp.save_npz(self.path + '/s_mean_adj_matrix.npz', mean_adj_matrix)
        return adj_matrix, norm_adj_matrix, mean_adj_matrix

    def creat_adj_matrix(self):
        """
        lil:基于行连接存储的稀疏矩阵(Row-based linked list sparse matrix)
        高效地添加、删除、查找元素 能够快速构建矩阵
        coo:(row_idx,col_idx),value
        csr:[每一行第1个元素在values中的idx] [col_idxs] [values]可以高效的进行矩阵运算
        """
        t0=time()
        adj_matrix=sp.dok_matrix((self.n_users+self.n_items,self.n_users+self.n_items),dtype=np.float32)
        adj_matrix=adj_matrix.tolil()
        R=self.R.tolil()

        adj_matrix[:self.n_users,self.n_users:]=R
        adj_matrix[self.n_users:,:self.n_users]=R.T
        adj_matrix=adj_matrix.todok()
        #print(adj_matrix)
        t1=time()
        print("create adjacency matrix,shpae:{},cost time:{:.1}s".format(adj_matrix.shape,t1-t0))

        # 归一化 相当于对于每个node 按照每一行的度进行归一化
        def normalized_adj(adj):
            rowsum = np.array(adj.sum(1)) # [N,1]
            d_inv = np.power(rowsum, -1).flatten() # 1/度 [N,]
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv) # 对角矩阵 [N,N]

            norm_adj = d_mat_inv.dot(adj) # 归一化操作 [N,N]
            #print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_matrix = normalized_adj(adj_matrix + sp.eye(adj_matrix.shape[0]))
        mean_adj_matrix = normalized_adj(adj_matrix)
        print("normalized adjacency matrix cost time:{:.1}s".format(time()-t1))
        return adj_matrix.tocsr(),norm_adj_matrix.tocsr(),mean_adj_matrix.tocsr()



from utils.parser import parse_args
args = parse_args()
data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
# 加载数据类 生成batch_data
data_generator=Data_Graph(data_path,args.batch_size)
A=data_generator.get_adj_matrix()[1].tocoo()
print(A.row.shape)
print(A.data.shape)
