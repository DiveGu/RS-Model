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
args=parse_args()
from utils.load_data import Data

data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)
loader=Data(data_path,1024)
loader.generate_train_cf_batch(0)
loader.generate_train_cf_batch(1)