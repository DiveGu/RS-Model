"""
- 2021/9/18

"""
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utils.helper import *
from time import time,strftime,localtime # 要用time()就不能import time了


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()

from collections import defaultdict
# 数据集准备
class Data_Sequence():

    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        self.train_file = path + '/train.csv'
        self.test_file = path + '/test.csv'

        ## 用户数量 物品数量 train数量
        #self.n_train, self.n_test = 0, 0
        #self.n_users, self.n_items = 0, 0

        #self.train_df, self.train_user_dict = self._load_ratings_train(train_file)
        #self.test_df, self.test_user_dict = self._load_ratings_test(test_file)
        ## train中的user集合
        #self.exist_users = self.train_user_dict.keys()

        self._create_dataset()

    def gen_neg(self,pos_list):
        neg_id=pos_list[0]
        while(neg_id in set(pos_list)):
            neg_id=random.randint(0,self.n_items-1)
        return neg_id

    def _create_dataset(self,max_len=50):
        train_df=pd.read_csv(self.train_file)
        test_df=pd.read_csv(self.test_file)
        self.n_items=max(train_df['item'].max(),test_df['item'].max())+1
        # 获取test pos id
        self.test_user_dict=dict(zip(test_df['user'],test_df['item']))

        train_data,val_data,test_data=defaultdict(list),defaultdict(list),defaultdict(list)
        for user_id,df in train_df[['user','item']].groupby('user'):
            pos_list=df['item'].tolist()
            neg_list=[self.gen_neg(pos_list+[self.test_user_dict[user_id]]) for _ in range(len(pos_list)-1+100)]
            # [.....val_id]
            for i in range(1,len(pos_list)):
                if(i==len(pos_list)-1):
                    val_data['hist'].append(pos_list[:i])
                    val_data['pos_id'].append(pos_list[i])
                    val_data['neg_id'].append(neg_list[i-1])
                else:
                    train_data['hist'].append(pos_list[:i])
                    train_data['pos_id'].append(pos_list[i])
                    train_data['neg_id'].append(neg_list[i-1])
            # test data
            test_data['hist'].append(pos_list)
            test_data['pos_id'].append(self.test_user_dict[user_id])
            test_data['neg_id'].append(neg_list[len(pos_list):])

        # 按照maxlen进行pad
        self.train = [pad_sequences(train_data['hist'], maxlen=maxlen,value=self.n_items), 
                      np.array(train_data['pos_id']),
                      np.array(train_data['neg_id'])]

        self.val = [pad_sequences(val_data['hist'], maxlen=maxlen,value=self.n_items), 
                    np.array(val_data['pos_id']),
                    np.array(val_data['neg_id'])]

        self.test = [pad_sequences(test_data['hist'], maxlen=maxlen,value=self.n_items),
                    np.array(test_data['pos_id']),
                    np.array(test_data['neg_id'])]

    # 生成训练batch
    def generate_train_batch(self,idx):
        #batch_num=self.train_df//self.batch_size
        if(idx==0):
            # 1 负采样
            #self.df_copy=self.train_df[['user','item']]
            #self.df_copy['neg_item']=self._sample()
            # 2 打乱数据
            state = np.random.get_state()
            for ar in self.train:
                np.random.set_state(state)
                np.random.shuffle(ar)

        # 3 生成batch数据
        start=idx*self.batch_size
        end=(idx+1)*self.batch_size
        end=end if end<self.n_train else self.n_train
        batch_data={
            'hist':self.train[0][start:end],
            'pos_id':self.train[1][start:end],
            'neg_id':self.train[2][start:end],
        }

        #return self.df_copy[start:end].values
        return batch_data 

    # 生成batch的feed_dict字典
    def generate_train_feed_dict(self,model,batch_data):
        feed_dict={
            model.hist:batch_data['hist'],
            model.pos_items:batch_data['pos_item'],
            model.neg_items:batch_data['neg_item']
        }
        
        return feed_dict

#data_gen=Data_Sequence('F:/data/experiment_data/ml-1m/5-core_tloo',256)

class SASRec():
    def __init__(self, args):
        self.model_type='SASRec'

        self.n_items=data_config['n_items']
        self.maxlen=args.maxlen
        self.layer_num=args.layer_num

        self.verbose=args.verbose

        self.emb_dim=args.embed_size
        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.hist=tf.placeholder(tf.int32,shape=[None,None],name='hist') # [N,max_len]
        self.pos_items=tf.placeholder(tf.int32,shape=[None,],name='pos_items') # [N,1]
        self.neg_items=tf.placeholder(tf.int32,shape=[None,],name='neg_items') # [N,1] or [N,4]

        # 初始化模型参数
        self.weights=self._init_weights()

        # 构造模型
        self._forward()

        # 初始化参数
        def _init_weights(self):
            all_weights=dict()
            initializer=tensorflow.contrib.layers.xavier_initializer()
            all_weights['item_embedding']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='item_embedding')
            print('using xavier initialization')        

            return all_weights

        # 构造模型
        def _forward(self):
            # 1 得到序列的表示
            his_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.hist) # [N,max_len,k]
            his_represention=self._bulid_his_represention(his_embeddings)
            # 2 得到pos和neg的target表示
            target_pos_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.pos_items) # [N,1,k]
            target_neg_embeddings=tf.nn.embedding_lookup(self.weights['item_embedding'],self.neg_items) # [N,4,k]
            # 3 得到预测评分
            pos_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_pos_embeddings))
            neg_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_neg_embeddings))
            self.batch_ratings=pos_preidct_scores
            # 4 构造损失函数
            batch_num=tf.dtypes.cast(tf.shape(neg_preidct_scores)[0], tf.int32)
            neg_num=tf.dtypes.cast(tf.shape(neg_preidct_scores)[1], tf.int32)
            cf_loss_list=[tf.math.log(pos_preidct_scores),tf.math.log(tf.ones(batch_num,neg_num)-neg_preidct_scores)]
            cf_loss=tf.reduce_mean(tf.concat(cf_loss_list,axis=1))
            # 5 优化

            return

        # 根据history items嵌入得到最终序列表示
        def _bulid_his_represention(self,his_embeddings):
            final_represention=tf.reduce_mean(his_embeddings,axis=1,keepdims=True) # [N,max_len,k] -> [N,1,k]
            return final_represention

        # 根据hist表示和target表示预测评分
        def _get_predict_score(self,hist_e,target_e):
            # [N,1,k],[N,4,k]
            predict_logit=tf.multiply(hist_e,target_e) # [N,4,k]
            predict_logit=tf.reduce_sum(predict_score,axis=2,keepdims=False) # [N,4,k] -> [N,4]

            return predict_logit