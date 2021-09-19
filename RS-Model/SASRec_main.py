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
parser.add_argument('--embed_size', default=20, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--regs',default='[1e-5,1e-5,1e-6]')

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

        #self._create_dataset()

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
        self.train = [pad_sequences(train_data['hist'], maxlen=max_len,value=self.n_items), 
                      np.array(train_data['pos_id']).reshape((-1,1)),
                      np.array(train_data['neg_id']).reshape((-1,1))]

        self.val = [pad_sequences(val_data['hist'], maxlen=max_len,value=self.n_items), 
                    np.array(val_data['pos_id']).reshape((-1,1)),
                    np.array(val_data['neg_id']).reshape((-1,1))]

        self.test = [pad_sequences(test_data['hist'], maxlen=max_len,value=self.n_items),
                    np.array(test_data['pos_id']).reshape((-1,1)),
                    np.array(test_data['neg_id']).reshape((-1,99))]

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
        end=end if end<self.train[0].shape[0] else self.train[0].shape[0]
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
            model.pos_items:batch_data['pos_id'],
            model.neg_items:batch_data['neg_id']
        }
        
        return feed_dict

    # 生成test的feed_dict字典
    def generate_test_feed_dict(self,model):
        feed_dict={
            model.hist:self.test[0],
            model.pos_items:np.concatenate((self.test[1],self.test[2]),axis=1)
        }
        
        return feed_dict
    
    # 统计训练集、验证集、测试集的数据量
    def print_data_info(self):
        print('train size:{}'.format(self.train[0].shape))
        print('val size:{}'.format(self.val[0].shape))
        print('test size:{}'.format(self.test[0].shape))


#data_gen=Data_Sequence('F:/data/experiment_data/ml-1m/5-core_tloo',256)

class SASRec():
    def __init__(self, args,data_config):
        self.model_type='SASRec'

        self.n_items=data_config['n_items']
        self.maxlen=args.maxlen

        self.emb_dim=args.embed_size
        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.hist=tf.placeholder(tf.int32,shape=[None,None],name='hist') # [N,max_len]
        self.pos_items=tf.placeholder(tf.int32,shape=[None,None],name='pos_items') # [N,1]
        self.neg_items=tf.placeholder(tf.int32,shape=[None,None],name='neg_items') # [N,1] or [N,4]

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
        pos_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_pos_embeddings)) # [N,1,k]
        neg_preidct_scores=tf.nn.sigmoid(self._get_predict_score(his_represention,target_neg_embeddings)) # [N,4,k]
        self.batch_ratings=tf.squeeze(pos_preidct_scores,axis=1)
        # 4 构造损失函数
        neg_num=tf.dtypes.cast(tf.shape(neg_preidct_scores)[1], tf.int32)
        cf_loss_list=[-tf.math.log(pos_preidct_scores),-tf.math.log(1-neg_preidct_scores)]
        cf_loss=tf.reduce_mean(tf.concat(cf_loss_list,axis=1))
        self.loss=cf_loss
        # 5 优化
        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        return

    # 根据history items嵌入得到最终序列表示
    def _bulid_his_represention(self,his_embeddings):
        final_represention=tf.reduce_mean(his_embeddings,axis=1,keepdims=True) # [N,max_len,k] -> [N,1,k]
        return final_represention

    # 根据hist表示和target表示预测评分
    def _get_predict_score(self,hist_e,target_e):
        # [N,1,k],[N,4,k]
        predict_logit=tf.multiply(hist_e,target_e) # [N,4,k]
        predict_logit=tf.reduce_sum(predict_logit,axis=2,keepdims=False) # [N,4,k] -> [N,4]

        return predict_logit

    # 训练
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss],feed_dict=feed_dict)

    # 预测
    def predict(self,sess,feed_dict):
        return sess.run([self.batch_ratings],feed_dict=feed_dict)

# 测试在test上的表现
def test_performance(predict,K):
    # predict [N,1+100] 
    hit=0.
    ndcg=0.
    # 看每一行的idx=0是否在topK中
    tmp=(-predict).argsort(axis=1)[:,:K]
    for i in range(tmp.shape[0]):
        for j in range(K):
            if(tmp[i][j]==0):
                hit+=1.
                ndcg+=1/np.log2(j+2)
                break
    hit/=tmp.shape[0]
    ndcg/=tmp.shape[0]

    return hit,ndcg


def main(args):
    # =================1：构造数据集===================
    t0=time()
    data_generator=Data_Sequence('F:/data/experiment_data/ml-1m/5-core_tloo',256)
    data_generator._create_dataset(args.maxlen)
    data_generator.print_data_info()
    t1=time()
    print('create dataset cost [{:.1f}s]'.format(t1-t0))
    # =================2：构造模型=====================
    data_config=dict()
    data_config['n_items']=data_generator.n_items
    model=SASRec(args,data_config)
    # =================3：训练模型=====================
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    loss_log,hit_log,ndcg_log=[],[],[]

    for epoch in range(args.epochs):
        t0=time()
        loss=0.
        batch_num=args.epochs//args.batch_size if args.epochs%args.batch_size==0 else args.epochs//args.batch_size+1
        for batch_idx in range(batch_num):
            batch_feed_data=data_generator.generate_train_batch(batch_idx)
            batch_feed_dict=data_generator.generate_train_feed_dict(model,batch_feed_data)
            _,batch_loss=model.train(sess,batch_feed_dict)
            loss+=batch_loss

        t1=time()
        loss=loss/batch_num
        loss_log.append(loss)
        show_loss_step=1
        show_val_step=5
        
        if((epoch+1)%show_loss_step==0):
            print('epoch:{}[{:.1f}s],loss:{:.5f}'.format(epoch,t1-t0,loss))
        if((epoch+1)%show_val_step==0):
            test_feed_dict=data_generator.generate_test_feed_dict(model)
            predict_score=model.predict(sess,test_feed_dict)
            predict_score=np.array(predict_score)
            print(type(predict_score))
            print(predict_score.shape)
            hit,ndcg=test_performance(predict_score,10)
            print('epoch:{},hit:{:.5f},ndcg:{:.5f}'.format(epoch,hit,ndcg))

main(args)

#test=[
#    [5,6,7,8,9,10,11,12,13],
#    [16,6,7,8,9,10,11,12,13],
#    [12,6,7,8,9,10,11,12,13],
#    ]
#test=np.array(test)

#test_performance(test,5)