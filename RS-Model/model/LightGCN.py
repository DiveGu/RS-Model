"""
- LightGCN
- 2021/7/18
"""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class LightGCN():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='LightGCN'
        self.pretrain_data=pretrain_data

        self.n_users=data_config['n_users']
        self.n_items=data_config['n_items']
        # 通过data_config传进来归一化邻接矩阵
        self.norm_adj = data_config['norm_adj']
        self.n_fold=100

        self.layer_num=args.layer_num

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

        self._forward()

        # 模型参数量
        self._static_params()

    # 构造模型
    def _forward(self):
        # 1 获取最终嵌入
        self.user_final_embeddings, self.item_final_embeddings = self._create_lightgcn_embed()
        # 2 查嵌入表获得u i表示
        u_e=tf.nn.embedding_lookup(self.user_final_embeddings,self.users) 
        pos_i_e=tf.nn.embedding_lookup(self.item_final_embeddings,self.pos_items)
        neg_i_e=tf.nn.embedding_lookup(self.item_final_embeddings,self.neg_items)
        # 3 预测评分
        self.batch_predictions=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,1]
        # 4 构造损失函数 优化
        self.mf_loss,self.reg_loss=self._creat_cf_loss(u_e,pos_i_e,neg_i_e)
        self.loss=self.mf_loss+self.reg_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    # 获取GCN最后的嵌入表
    def _create_lightgcn_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)
        
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0) # [u+i,k]
        all_embeddings = [ego_embeddings]
        
        # 进行多层GCN
        for k in range(0, self.layer_num):

            temp_embed = []
            # 原始：L*H:[N,N] [N,K] -> [N,K]
            # 分块：将L[N,N] 分成了 [N//n_fold,N],[N//n_fold,N],...,[N-[N//n_fold*(n_fold-1)],N]
            for f in range(self.n_fold):
                # [N//n_fold,N] [N,K] -> [N//n_fold,K]
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0) # 按照行拼起来还是 [N,K]
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings=tf.stack(all_embeddings,1) # tensor拼接 [N,K] -> [N,1+layer_num,K]
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False) # 平均所有嵌入表 [N,K]
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0) # H拆成U I
        return u_g_embeddings, i_g_embeddings

    # 将邻接矩阵分成fold块
    def _split_A_hat(self, X):
        """
        X:一个稀疏矩阵
        return 分成fold块[每一块矩阵是原矩阵的几行]的稀疏矩阵list
        """
        A_fold_hat = []

        # 按整行来对于矩阵分块 将N行的矩阵分成fold块
        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

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