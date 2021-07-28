"""
- DGCF
- 2021/7/28
"""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class DGCF():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='DGCF'
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

        # factor数量
        self.factor_num=args.factor_num
        self.factor_dim=self.emb_dim//self.factor_num
        self.factor_dim=self.factor_num*self.factor_dim # 保证emb_dim/factor_num==factor_dim

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
    
    # 模型主体架构 在每个factor上按照score进行动态路由
    def _factor_dynamic_routing(self):
        # 拼接u i 获取所有节点的整个嵌入表
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0) # [U+I,k]
        all_embeddings = [ego_embeddings]

        # 做L层卷积
        for i in range(self.layer_num):
            # 把整个嵌入划分成factor_num个子嵌入表
            ego_factor_emb_list=tf.split(all_embeddings[-1])

            break


        return



    def _create_star_routing_embed_with_P(self, pick_ = False):
        '''
        pick_ : True, the model would narrow the weight of the least important factor down to 1/args.pick_scale.
        pick_ : False, do nothing.
        '''
        p_test = False
        p_train = False

        A_values = tf.ones(shape=[self.n_factors, len(self.all_h_list)])
        # get a (n_factors)-length list of [n_users+n_items, n_users+n_items]

        # load the initial all-one adjacency values
        # .... A_values is a all-ones dense tensor with the size of [n_factors, all_h_list].
        

        # get the ID embeddings of users and items
        # .... ego_embeddings is a dense tensor with the size of [n_users+n_items, embed_size];
        # .... all_embeddings stores a (n_layers)-len list of outputs derived from different layers.
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        all_embeddings_t = [ego_embeddings]

        output_factors_distribution = []
        
        factor_num = [self.n_factors, self.n_factors, self.n_factors]
        iter_num = [self.n_iterations, self.n_iterations, self.n_iterations]
        for k in range(0, self.n_layers):
            # prepare the output embedding list
            # .... layer_embeddings stores a (n_factors)-len list of outputs derived from the last routing iterations.
            n_factors_l = factor_num[k]
            n_iterations_l = iter_num[k]
            layer_embeddings = []
            layer_embeddings_t = []
            
            # split the input embedding table
            # .... ego_layer_embeddings is a (n_factors)-leng list of embeddings [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = tf.split(ego_embeddings, n_factors_l, 1)
            ego_layer_embeddings_t = tf.split(ego_embeddings, n_factors_l, 1) 

            # perform routing mechanism
            for t in range(0, n_iterations_l):
                iter_embeddings = []
                iter_embeddings_t = []
                A_iter_values = []

                # split the adjacency values & get three lists of [n_users+n_items, n_users+n_items] sparse tensors
                # .... A_factors is a (n_factors)-len list, each of which is an adjacency matrix
                # .... D_col_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. columns
                # .... D_row_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. rows
                if t == n_iterations_l - 1:
                    p_test = pick_
                    p_train = False

                A_factors, D_col_factors, D_row_factors = self._convert_A_values_to_A_factors_with_P(n_factors_l, A_values, pick= p_train)
                A_factors_t, D_col_factors_t, D_row_factors_t = self._convert_A_values_to_A_factors_with_P(n_factors_l, A_values, pick= p_test)
                for i in range(0, n_factors_l):
                    # update the embeddings via simplified graph convolution layer
                    # .... D_col_factors[i] * A_factors[i] * D_col_factors[i] is Laplacian matrix w.r.t. the i-th factor
                    # .... factor_embeddings is a dense tensor with the size of [n_users+n_items, embed_size/n_factors]
                    factor_embeddings = tf.sparse.sparse_dense_matmul(D_col_factors[i], ego_layer_embeddings[i])
                    factor_embeddings_t = tf.sparse.sparse_dense_matmul(D_col_factors_t[i], ego_layer_embeddings_t[i])

                    factor_embeddings_t = tf.sparse.sparse_dense_matmul(A_factors_t[i], factor_embeddings_t)
                    factor_embeddings = tf.sparse.sparse_dense_matmul(A_factors[i], factor_embeddings)

                    factor_embeddings = tf.sparse.sparse_dense_matmul(D_col_factors[i], factor_embeddings)
                    factor_embeddings_t = tf.sparse.sparse_dense_matmul(D_col_factors_t[i], factor_embeddings_t)

                    iter_embeddings.append(factor_embeddings)
                    iter_embeddings_t.append(factor_embeddings_t)
                    
                    if t == n_iterations_l - 1:
                        layer_embeddings = iter_embeddings
                        layer_embeddings_t = iter_embeddings_t

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    head_factor_embedings = tf.nn.embedding_lookup(factor_embeddings, self.all_h_list)
                    tail_factor_embedings = tf.nn.embedding_lookup(ego_layer_embeddings[i], self.all_t_list)

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    head_factor_embedings = tf.math.l2_normalize(head_factor_embedings, axis=1)
                    tail_factor_embedings = tf.math.l2_normalize(tail_factor_embedings, axis=1)

                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [all_h_list,1]
                    A_factor_values = tf.reduce_sum(tf.multiply(head_factor_embedings, tf.tanh(tail_factor_embedings)), axis=1)

                    # update the attentive weights
                    A_iter_values.append(A_factor_values)

                # pack (n_factors) adjacency values into one [n_factors, all_h_list] tensor
                A_iter_values = tf.stack(A_iter_values, 0)
                # add all layer-wise attentive weights up.
                A_values += A_iter_values
                
                if t == n_iterations_l - 1:
                    #layer_embeddings = iter_embeddings
                    output_factors_distribution.append(A_factors)

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = tf.concat(layer_embeddings, 1)
            side_embeddings_t = tf.concat(layer_embeddings_t, 1)
            
            ego_embeddings = side_embeddings
            ego_embeddings_t = side_embeddings_t
            # concatenate outputs of all layers
            all_embeddings_t += [ego_embeddings_t]
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        all_embeddings_t = tf.stack(all_embeddings_t, 1)
        all_embeddings_t = tf.reduce_mean(all_embeddings_t, axis=1, keep_dims=False)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        u_g_embeddings_t, i_g_embeddings_t = tf.split(all_embeddings_t, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings, output_factors_distribution, u_g_embeddings_t, i_g_embeddings_t

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
