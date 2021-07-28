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
        self.pool_type='mean_atten'

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
        # 2-1 根据pos_target_item得到得嵌入
        # 2-2 根据neg_target_item得到得嵌入
        if(self.pool_type=='mean'):
            p_embeddings_pos=self._his_pool(his_e,'mean')
            p_embeddings_neg=p_embeddings_pos
        elif(self.pool_type=='sum'):
            p_embeddings_pos=self._his_pool(his_e,'sum')
            p_embeddings_neg=p_embeddings_pos
        elif(self.pool_type=='target_atten'):
            p_embeddings_pos = self._attention(his_e,pos_i_e,his_e)
            p_embeddings_neg = self._attention(his_e,neg_i_e,his_e)
        elif(self.pool_type=='mean_atten'):
            p_embeddings_pos = self._attention(his_e,self._his_pool(his_e,'mean'),his_e)
            p_embeddings_neg=p_embeddings_pos

        # 3 预测评分
        #self.batch_predictions=tf.reduce_sum(tf.multiply(p_embeddings,pos_i_e),axis=1) # [N,1]
        pos_scores=self._get_score(p_embeddings_pos,pos_i_e)
        neg_scores=self._get_score(p_embeddings_neg,neg_i_e)
        self.batch_predictions=pos_scores
        # 4 构造损失函数 优化
        self.mf_loss,self.reg_loss=self._creat_cf_loss(pos_scores,neg_scores,[his_e,pos_i_e,neg_i_e])
        self.loss=self.mf_loss+self.reg_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _attention(self,key,query,value):
        """
        # 历史序列中和target item越接近的 权重越高
        key=[N,m,k]
        query=[N,k]
        value=[N,m,k]
        最终得到 [N,k]
        """
        query=tf.expand_dims(query,1) # [N,k] -> [N,1,k]
        score_logit=key*query # [N,m,k]*[N,1,k]->[N,m,k]
        score_logit=tf.reduce_sum(score_logit,axis=2,keepdims=False) # [N,m]
        
        # softmax 关键点是不能计算序列中填充的item score_f
        mask_mat=self._get_binary_tensor(self.user_his_item,self.n_items)
        mask_mat=tf.cast(mask_mat,tf.float32) # [N,m] 填充的itemid对应为0 真实的为1
        
        # 写法1 自己写softmax
        score_logit_exp=tf.exp(score_logit)*mask_mat # [N,m] 填充对应位置exp=0
        score_logit_exp_sum=tf.reduce_sum(score_logit_exp,axis=1,keepdims=True) # [N,1] 这样求和项忽略掉了填充值

        score_atten=score_logit_exp*tf.pow(score_logit_exp_sum,-1) # [N,m] / [N,1] -> [N,m]
        
        # 根据score和value聚合得结果
        score_atten=tf.expand_dims(score_atten,2) # [N,m,1]


        return tf.reduce_sum(score_atten*value,axis=1,keepdims=False) # [N,m,k] -> [N,k]

    # 通过mean或sum聚合item序列
    def _his_pool(self,his_e,pool_type='mean'):
        """
        his_e:[N,m,k]
        his_pool:[N,k]
        """
        mask_mat=self._get_binary_tensor(self.user_his_item,self.n_items) # [N,m]
        mask_mat=tf.expand_dims(tf.cast(mask_mat,tf.float32),2) # [N,m,1] 填充的itemid对应为0 真实的为1
        his_pool=tf.reduce_sum(his_e*mask_mat,axis=1,keepdims=False) # [N,m,k] -> [N,k]
        if(pool_type=='mean'):
            his_item_num=tf.reduce_sum(mask_mat,axis=1,keepdims=False) # [N,1]
            his_pool=his_pool*tf.pow(his_item_num,-1) # [N,k] * [N,1]

        return his_pool


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
            #all_weights['user_embedding']=tf.Variable(initializer([self.n_users,self.emb_dim]),name='user_embedding')
            all_weights['item_embedding_history']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='item_embedding_history')
            all_weights['item_embedding_target']=tf.Variable(initializer([self.n_items+1,self.emb_dim]),name='item_embedding_target')
            #all_weights['item_embedding_target']=all_weights['item_embedding_history']
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
