"""
- LightGCN
- 2021/7/18
"""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class NAIS():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='LightGCN'
        self.pretrain_data=pretrain_data

        self.n_users=data_config['n_users']
        self.n_items=data_config['n_items']

        """
        1 如果直接用R的话 不能用lookup来查每个uid 内存不够 通过data_config传进来训练集R
        2 用序列表示每个uid 固定长度 不足的进行padding
        """
        #self.R = data_config['r_matrix']

        self.verbose=args.verbose

        self.emb_dim=args.embed_size
        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.users=tf.placeholder(tf.int32,shape=[None,],name='users')
        self.user_his_item=tf.placeholder(tf.int32,shape=[None,None],name='user_his_item')
        self.pos_items=tf.placeholder(tf.int32,shape=[None,],name='pos_items')
        self.neg_items=tf.placeholder(tf.int32,shape=[None,],name='neg_items')

        # 初始化模型参数
        self.weights=self._init_weights()

        self._forward()

        # 模型参数量
        self._static_params()

    # 构造模型
    def _forward(self):
        # 1 查嵌入表获得正负item作为traget的表示
        pos_i_e=tf.nn.embedding_lookup(self.weights['item_embedding_target'],self.pos_items) # [N,1]
        neg_i_e=tf.nn.embedding_lookup(self.weights['item_embedding_target'],self.neg_items) # [N,1]
        # 2 根据user的历史记录和target item 获取user最终嵌入
        his_e=tf.nn.embedding_lookup(self.weights['item_embedding_history'],self.user_his_item) # [N,m,K] 假设序列长度为m
        p_embeddings = self._attention(his_e)
        # 3 预测评分
        self.batch_predictions=tf.reduce_sum(tf.multiply(p_embeddings,pos_i_e),axis=1) # [N,1]
        # 4 构造损失函数 优化
        self.mf_loss,self.reg_loss=self._creat_cf_loss(p_embeddings,pos_i_e,neg_i_e)
        self.loss=self.mf_loss+self.reg_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _attention(self,p):
        return tf.reduce_mean(p,axis=1,keepdims=False)


    # 将X转化成稀疏矩阵
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        X:sp的矩阵
        """
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose() # [2,N] -> [N,2]
        # 创建稀疏矩阵、indices是index[row col]、data是值
        # 注：加上shape 因为可能有全0行 全0列 所以coo形式必须加shape
        return tf.SparseTensor(indices, coo.data, coo.shape)

    # 初始化模型参数
    def _init_weights(self):
        all_weights=dict()

        initializer=tensorflow.contrib.layers.xavier_initializer()
        if(self.pretrain_data==None):
            #all_weights['user_embedding']=tf.Variable(initializer([self.n_users,self.emb_dim]),name='user_embedding')
            all_weights['item_embedding_history']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='item_embedding_history')
            all_weights['item_embedding_target']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='item_embedding_target')
            print('using xavier initialization')        
        else:
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
