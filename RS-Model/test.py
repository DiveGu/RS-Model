import numpy as np
import pandas as pd
from utils.parser import parse_args

#pre='15-filter'
#print(pre.split('-'))

#a=set([3])
#b=set([1,2,5,6])
#result=list(a|b)

#print(a)
#print(b)
#print(result)

#a,b = np.unique(np.array([3,4,5,1,2,4,4,4]), return_inverse=True)
#print(a)
#print(b)

#df=pd.DataFrame(data=np.arange(12).reshape(4,3),columns=['a','b','c'])
#print(df.loc[0:2,:]) #注意能取到 0/1/2行
#print(df.iloc[0:2,:]) #注意能取到 0/1行

#print(list(range(1,1))) # nul


#print(np.random.choice([1,2,3],0))


# random从set里随机选
#import random
#print(random.sample({1,2,3,4,5,6,7,8,9}, 5))

# 读取保存的txt 中的dict
#import json
#f = open('F:/data/experiment_data/lastfm/test_neg.txt','r')
#a=f.read()
#mdict=json.loads(a)
#f.close()
##print(type(mdict))
##print(type(mdict['0']))
#for i in range(100):
#    print(i,len(mdict[str(i)]))
##print(mdict['0'])

# 测试load_data
#args=parse_args()
#from utils.load_data import Data

#data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
#loader=Data(data_path,1024)
#loader.generate_train_cf_batch(0)
#loader.generate_train_cf_batch(1)


# 测试tf.reduce_mean
#import tensorflow.compat.v1 as tf
 
#x = [[1,2,3],
#      [1,2,3]]
 
#xx = tf.cast(x,tf.float32)
 
#mean_all = tf.reduce_mean(xx, keep_dims=False)
#mean_0 = tf.reduce_mean(xx, axis=0, keep_dims=False)
#mean_1 = tf.reduce_mean(xx, axis=1, keep_dims=False)
 
 
#with tf.Session() as sess:
#    m_a,m_0,m_1 = sess.run([mean_all, mean_0, mean_1])
 
#print (m_a)    # output: 2.0
#print (m_0)    # output: [ 1.  2.  3.]
#print (m_1)    # output:  [ 2.  2.]


#print (xx.shape)
#print (m_a.shape)
#print (m_0.shape)
#print (m_1.shape)


# 测试array的一些操作 主要是metrics


# 测试zip(np.array list)
#A=np.array(np.arange(0,50).reshape(10,5)) # [10,5]
#B=list(range(10)) #[10]
#print(A)
#print(B)
#C=zip(A,B)
#print(type(C))
#for a in C:
#    print(type(a))


# 测试 np.array(list)的shape
#a=[1,2,3,4,5]
#a=np.array(a)
#print(a)
#print(a.shape)



## 测试字典.get是什么类型
#keys=[1,2,3,4]
#values=[5,4,3,2]
#mdict=dict(zip(keys,values))

#print(type(mdict.get))
#print(mdict.get)

## 测试heapq 对字典排序 key 
#import heapq

#def sort_key(x):
#    return x[1],x[0]

##ret=heapq.nlargest(2, mdict, key=mdict.get) # 按照v排序 返回k
#ret=heapq.nlargest(2, mdict.items(), key=lambda x:x[1]) # ok 返回排序的元组
##ret=heapq.nlargest(2, mdict) # 返回的结果没看懂 既不是key排序 也不是value排序
#print(ret)


## 测试list转str
#list1=[1.789,1.566,2.456]
#ret=" ".join([str(x)[:4] for x in list1])
#print(ret)


# 测试np.array转list

#arr1=np.array([1.56,5,67,7.89])


#def convert_list_2_str(lst,num):
#    return " ".join([str(x)[:num] for x in list(lst)])

#ret=convert_list_2_str(arr1,3)
#print(ret)


# 测试遍历np.array
#arr1=np.array([1.2,3.4,5,6,7,8.9])
#for x in arr1:
#    print(x)

## 测试list 元素是np.array 转str
#arr1=np.array(range(0,5))
#arr2=np.array(range(1,6))
#arr3=np.array(range(2,7))
#arr4=np.array(range(3,8))

#lst=[arr1,arr2,arr3,arr4]

#lst_str=str([x.tolist() for x in lst])
#print(lst_str)


# 测试获取时间str
#import time
#from time import strftime,localtime
#cur=strftime("%Y-%m-%d %H:%M:%S", localtime()) 
#print(type(cur))
#print(cur)


# 测试 [1,2,3,4]*3
#lst1=[1,2,3]
#print(lst1*3)


# 测试 zip(数量不一样)

#a1=list(range(100))
#a1_new=np.array(a1).reshape(-1,10)
#a2=list(range(10))

#for x,y in zip(a1_new,a2):
#    print(x)
#    print(y)


## 测试tf中+或- [5,1]和[5,5]
#import tensorflow.compat.v1 as tf
 
## [2,2]
#x = [[1,2],
#      [3,4]]

## [2,1]
#y=[[7],[8]]
 
#x1 = tf.cast(x,tf.float32)
#x2= tf.cast(y,tf.float32)
 
#result=x1-x2

#with tf.Session() as sess:
#    ret = sess.run([result])
 
#print (ret)  


# 测试测试tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=X_logit)
#import tensorflow.compat.v1 as tf
#label=[[0,0,1],
#       [1,0,0],
#       [0,1,0]
#    ]

#X_logit=[[0,0,10],
#       [10,8,1],
#       [5,8,2]
#    ]

#label=tf.cast(label,tf.float32)
#X_logit=tf.cast(X_logit,tf.float32)

#result=tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=X_logit)


#with tf.Session() as sess:
#    ret = sess.run([result])
 
#print (ret)  

# 测试tf.tile
# label = tf.tile(tf.eye(self.factor_num), [n_size, 1])

#import tensorflow.compat.v1 as tf


#label=tf.tile(tf.eye(5), [3, 3])


#with tf.Session() as sess:
#    ret = sess.run([label])
 
#print (ret)  


# 测试 tf.sequence_mask()

#import tensorflow.compat.v1 as tf

#x=tf.sequence_mask([[1, 3, 2],
#                    [0, 2, 3]],3,dtype = tf.float32) # [N,k] 变成 [N,k,max_len]
#"""
#最里面增加1维，维度是max_len
#取值2 表示 位置 1、2为True 位置3为Fasle
#"""

#with tf.Session() as sess:
#    ret = sess.run([x])

##[array([[[ True, False, False],
##        [ True,  True,  True],
##        [ True,  True, False]],

##       [[False, False, False],
##        [ True,  True, False],
##        [ True,  True,  True]]])]
 
#print (ret)  


## 测试list转set 的元素顺序
#ret=[2,2,3,3,5,5,1,1,4,4,4,]
#print(set(ret))


# 测试tensor类型 x[1] 取一行 一列 
# 可以直接取
#import tensorflow.compat.v1 as tf
#X=[[0,0,1,2],
#       [1,0,0,3],
#       [0,1,0,4]
#    ]

#X=tf.cast(X,tf.float32)
#print(X[0].shape)


#with tf.Session() as sess:
#    ret = sess.run([X[:,3]])
 
#print (ret)  

#with tf.Session() as sess:
#    ret = sess.run([X[0]])
 
#print (ret)  


## 测试tf.SpareTensor
#import tensorflow.compat.v1 as tf
## 5条边 [4,4]的形状 6个factor
#A_indices=[
#    [0,0],
#    [0,2],
#    [1,2],
#    [2,1],
#    [2,2]
#    ]
#D_indices=[
#    [0,1,2,3],
#    [0,1,2,3],
#    ]
#D_indices=np.mat(D_indices).transpose()

#A_factor_values=tf.ones(shape=[6,5])
#A_factor_scores=tf.nn.softmax(A_factor_values,axis=0)

#for i in range(0, 6):
#    # 【2-1】:提取factor i 对应的score行 并转化为稀疏tensor
#    A_i_scores = A_factor_scores[i] # 1维 [Edge]
#    print('一个factor所有边的A score:{}'.format(A_i_scores.shape))
#    # [Edge,2] [Edge] () -> tensor [U+I,U+I]
#    A_i_tensor = tf.SparseTensor(A_indices, A_i_scores, [4,4])
#    print('一个factor所有边的A score 稀疏tensor:{}'.format(A_i_tensor.shape))

#    # 【2-2】:计算每个factor下的拉普拉斯矩阵
#    # 计算当前factor i的score的度（准确的说，是sum A value）
#    # 分别是：求每一行的和 求每一列的和
#    D_i_col_scores_sum = tf.sparse_reduce_sum(A_i_tensor, axis=1,keepdims=False)
#    print('D_i_col_scores_sum:{}'.format(D_i_col_scores_sum.shape))
#    D_i_col_scores = 1/tf.math.sqrt(D_i_col_scores_sum) # [U+I,]
#    print('D_i_col_scores:{}'.format(D_i_col_scores.shape))
         
#    # 【2-3】:重新修正两个Laplace 稀疏tensor 的形状
#    D_i_col_tensor = tf.SparseTensor(D_indices, D_i_col_scores, [4,4]) # [U+I,U+I]
#    print('D_i_col_tensor:{}'.format(D_i_col_tensor.shape))

#    break


#with tf.Session() as sess:
#    ret_lst = sess.run([A_i_scores,A_i_tensor,D_i_col_scores_sum,D_i_col_scores,D_i_col_tensor])
 
#for ret in ret_lst:
#    print('****************************')
#    print(ret)

# 测试np.ones_like(10)
print(np.ones_like(10))



