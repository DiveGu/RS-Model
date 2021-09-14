"""
- BPRMF
- 2021/5/9
"""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class BPRMF():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='mf'
        self.pretrain_data=pretrain_data

        self.n_users=data_config['n_users']
        self.n_items=data_config['n_items']

        self.verbose=args.verbose

        self.emb_dim=args.embed_size
        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.users=tf.placeholder(tf.int32,shape=[None,],name='users')
        self.pos_items=tf.placeholder(tf.int32,shape=[None,],name='pos_items')
        self.neg_items=tf.placeholder(tf.int32,shape=[None,],name='neg_items')

        # 初始化模型参数
        self.weights=self._init_weights()

        # 查嵌入表获得表示
        #self.weights['user_embedding']=tf.math.l2_normalize(self.weights['user_embedding'], axis=1)
        #self.weights['item_embedding']=tf.math.l2_normalize(self.weights['item_embedding'], axis=1)
        u_e=tf.nn.embedding_lookup(self.weights['user_embedding'],self.users) # 
        pos_i_e=tf.nn.embedding_lookup(self.weights['item_embedding'],self.pos_items)
        neg_i_e=tf.nn.embedding_lookup(self.weights['item_embedding'],self.neg_items)

        #u_e=tf.math.l2_normalize(u_e,axis=1)
        #pos_i_e=tf.math.l2_normalize(pos_i_e,axis=1)
        #neg_i_e=tf.math.l2_normalize(neg_i_e,axis=1)

        # 预测评分
        #self.batch_predictions=tf.matmul(u_e,pos_i_e,transpose_b=True)
        self.batch_predictions=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,1]

        # 构造损失函数 优化
        self.mf_loss,self.reg_loss=self._creat_cf_loss(u_e,pos_i_e,neg_i_e)
        self.loss=self.mf_loss+self.reg_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # 模型参数量
        self._static_params()


    # 初始化模型参数
    def _init_weights(self):
        all_weights=dict()

        initializer=tensorflow.contrib.layers.xavier_initializer()
        if(self.pretrain_data==None):
            all_weights['user_embedding']=tf.Variable(initializer([self.n_users,self.emb_dim]),name='user_embedding')
            all_weights['item_embedding']=tf.Variable(initializer([self.n_items,self.emb_dim]),name='item_embedding')
            print('using xavier initialization')        
        else:
            all_weights['user_embedding']=tf.Variable(initial_value=self.pretrain_data['user_embed'],trainable=True,
                                                      name='user_embedding',dtype=tf.float32)
            all_weights['item_embedding']=tf.Variable(initial_value=self.pretrain_data['item_embed'],trainable=True,
                                                      name='item_embedding',dtype=tf.float32)
            print('using pretrained user/item embeddings')
        return all_weights

    # 构造cf损失函数
    def _creat_cf_loss(self,u_e,pos_i_e,neg_i_e):

        pos_scores=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,1,K] [N,1,K] -> [N,1,K] -> [N,1]
        neg_scores=tf.reduce_sum(tf.multiply(u_e,neg_i_e),axis=1) # [N,1,K] [N,1,K] -> [N,1,K] -> [N,1]

        regular=tf.nn.l2_loss(u_e)+tf.nn.l2_loss(pos_i_e)+tf.nn.l2_loss(neg_i_e) # 1

        diff=tf.log(tf.nn.sigmoid(pos_scores-neg_scores)) # [N,1]

        mf_loss=-(tf.reduce_mean(diff)) # [N,1] -> 1
        reg_loss=self.regs[0]*regular

        return mf_loss,reg_loss


    # 统计参数量
    def _static_params(self):
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    # train
    def train(self,sess,feed_dict):
        return sess.run([self.opt,self.loss,self.mf_loss,self.reg_loss],feed_dict)

    # predict
    def predict(self,sess,feed_dict):
        batch_predictions=sess.run(self.batch_predictions,feed_dict)
        return batch_predictions

    # save learned embeddings
    def save_tensor(self,sess,path):
        user_embed, item_embed = sess.run(
            [self.weights['user_embedding'], self.weights['item_embedding']],
            feed_dict={})

        np.savez(path, user_embed=user_embed, item_embed=item_embed)
        print('save the weights of fm in path: ', path)
