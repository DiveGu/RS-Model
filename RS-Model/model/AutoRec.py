"""
- AutoRec
- 2021/10/22
"""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class AutoRec():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='AutoRec'
        self.pretrain_data=pretrain_data

        self.n_users=data_config['n_users']
        self.n_items=data_config['n_items']

        # u2i 稀疏矩阵 相当于可以通过uid获取其所有train的行为序列
        self.R = data_config['r_matrix']

        self.verbose=args.verbose

        self.q_dims=args.q_dims
        self.p_dims=args.p_dims

        self.emb_dim=self.q_dims[0]
        self.lr=args.lr

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.users=tf.placeholder(tf.int32,shape=[None,],name='users')
        self.pos_items=tf.placeholder(tf.int32,shape=[None,],name='pos_items')
        self.neg_items=tf.placeholder(tf.int32,shape=[None,],name='neg_items')

        # 初始化模型参数
        self.weights=self._init_weights()

        self._forward()

        # 模型参数量
        self._static_params()

    # 构造模型
    def _forward(self):
        # 1 Encoder 编码得Z

        # 2 Decoder 得重构X

        # 3 loss: |x-x^|

        return



    def _encoder(self,inputs):


        return

    def _decoder(self):


        return


    # 计算预测得分score_ui
    def _get_score(self,user_emb,item_emb):
        score_cf=tf.multiply(user_emb,item_emb)
        score_cf=tf.reduce_sum(score_cf,axis=1,keepdims=False)

        return score_cf
    
    def _get_binary_tensor(self,tensor, max_len):
        one = tf.ones_like(tensor)
        zero = tf.zeros_like(tensor)
        return tf.where(tensor < max_len, one, zero)

    def _attention_MLP(self, q_):
       with tf.name_scope("attention_MLP"):
            b = tf.shape(q_)[0] # batch内用户数量
            n = tf.shape(q_)[1] # 序列长度
            r = (self.algorithm + 1)*self.embedding_size # 嵌入维度

            MLP_output = tf.matmul(tf.reshape(q_,[-1,r]), self.W) + self.b #(b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            if self.activation == 0:
                MLP_output = tf.nn.relu( MLP_output )
            elif self.activation == 1:
                MLP_output = tf.nn.sigmoid( MLP_output )
            elif self.activation == 2:
                MLP_output = tf.nn.tanh( MLP_output )

            A_ = tf.reshape(tf.matmul(MLP_output, self.h),[b,n]) #(b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(A_) # [b,n]
            num_idx = tf.reduce_sum(self.num_idx, 1) # [b,]
            mask_mat = tf.sequence_mask(num_idx, maxlen = n, dtype = tf.float32) # [b,] -> [b,n]
            exp_A_ = mask_mat * exp_A_ # [b,n]
            exp_sum = tf.reduce_sum(exp_A_, 1, keep_dims=True)  # (b, 1) 这样保证了sum里面没有pad的值
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1])) # [b,1]

            A = tf.expand_dims(tf.div(exp_A_, exp_sum),2) # (b, n, 1)

            return tf.reduce_sum(A * self.embedding_q_, 1)     


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
            all_weights['item_embedding_q']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='item_embedding_q')
            all_weights['item_embedding_p']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='item_embedding_p')
            print('using xavier initialization')        
        else:
            all_weights['item_embedding']=tf.Variable(initial_value=self.pretrain_data['item_embed'],trainable=True,
                                                      name='item_embedding',dtype=tf.float32)

            print('using pretrained user/item embeddings')

        return all_weights

    # 构造cf损失函数
    def _creat_cf_loss(self,pos_scores,neg_scores,reg_list):
        regular=0.
        for e in reg_list:
            regular+=tf.nn.l2_loss(e)

        diff=tf.log(1e-12+tf.nn.sigmoid(pos_scores-neg_scores)) # [N,]

        mf_loss=-(tf.reduce_mean(diff)) # [N,] -> 1
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

