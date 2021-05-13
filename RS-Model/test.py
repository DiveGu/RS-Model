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
