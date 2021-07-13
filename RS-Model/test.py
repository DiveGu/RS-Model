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

a1=list(range(100))
a1_new=np.array(a1).reshape(-1,10)
a2=list(range(10))

for x,y in zip(a1_new,a2):
    print(x)
    print(y)
