"""
- DGCF
- 2021/7/28
"""
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class GNUD():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='GNUD'
        self.pretrain_data=pretrain_data

        self.n_users=data_config['n_users']
        self.n_items=data_config['n_items']
        # 通过data_config传进来邻接矩阵
        # 需要提取将邻接矩阵A转化成coo [row,col,data] h即为row t即为col
        self.norm_adj = data_config['norm_adj'] # 邻接矩阵
        self.all_h_list = data_config['all_h_list'] # 所有Edge的头节点 list
        self.all_t_list = data_config['all_t_list'] # 所有Edge的尾节点 list
        self.A_in_shape = self.norm_adj.tocoo().shape # 邻接矩阵的形状

        self.factor_class_layers=eval(args.factor_class_layers)

        self.n_fold=100
        self.n_iteration=args.n_iteration

        self.layer_num=args.layer_num

        self.verbose=args.verbose

        self.emb_dim=args.embed_size
        self.lr=args.lr

        # factor数量
        self.factor_num=args.factor_num
        self.factor_dim=args.factor_dim

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

    # 构造模型
    def _forward(self):
        # 1 把user item嵌入分别投影到factor的空间中
        self.user_factor_emb_list=self._get_disen_embedding(self.weights['user_embedding'])
        self.item_factor_emb_list=self._get_disen_embedding(self.weights['item_embedding'])

        # 2 获取最终嵌入
        # 2-1 L层卷积+邻居路由
        node_gcn_embeddings,A_scores_tensor = self._factor_dynamic_routing()     
        # 2-2 MLP得最终嵌入
        node_gcn_embeddings=tf.layers.dense(node_gcn_embeddings,
                                        self.emb_dim,
                                        use_bias=True,
                                        kernel_regularizer=tf.keras.regularizers.l2(self.regs[2]),
                                        bias_regularizer=tf.keras.regularizers.l2(self.regs[2]),
                                        activation='relu',
                                        name="final_embedding_layer",
                                        reuse=tf.AUTO_REUSE,
                                        ) # [U+I,emb_dim]
        
        self.user_final_embeddings,self.item_final_embeddings=tf.split(node_gcn_embeddings,[self.n_users,self.n_items],axis=0)

        # 3 查嵌入表获得u i表示
        u_e=tf.nn.embedding_lookup(self.user_final_embeddings,self.users) 
        pos_i_e=tf.nn.embedding_lookup(self.item_final_embeddings,self.pos_items)
        neg_i_e=tf.nn.embedding_lookup(self.item_final_embeddings,self.neg_items)
        # 4 预测评分
        self.batch_predictions=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,1]
        # 5 构造cf损失函数
        self.mf_loss,self.reg_loss=self._creat_cf_loss(u_e,pos_i_e,neg_i_e)
        # 6 构造factor独立性损失函数
        if(self.regs[1]<1e-5):
            self.factor_loss=tf.constant(0.0)
        else:
            self.factor_loss=self._creat_factor_loss_cos(tf.split(self.user_final_embeddings,self.factor_num,axis=1),self.users)+\
                                    self._creat_factor_loss_cos(tf.split(self.item_final_embeddings,self.factor_num,axis=1),self.pos_items)+\
                                    self._creat_factor_loss_cos(tf.split(self.item_final_embeddings,self.factor_num,axis=1),self.neg_items)

        self.loss=self.mf_loss+self.reg_loss+self.factor_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

 
    # 模型主体架构 在每个factor上按照score进行动态路由 最后接一个MLP
    def _factor_dynamic_routing(self):
        #self.user_factor_emb_list
        user_factor_concat_embedding=tf.concat(self.user_factor_emb_list,axis=1) # [U,factor_dim] -> [U,emb_dim]
        item_factor_concat_embedding=tf.concat(self.item_factor_emb_list,axis=1) # [I,factor_dim] -> [I,emb_dim]
        # 拼接u i 获取所有节点的整个嵌入表
        ego_embeddings = tf.concat([user_factor_concat_embedding, item_factor_concat_embedding], axis=0) # [U+I,k]
        all_embeddings = [ego_embeddings]

        # 【step 1】：做L层卷积 每层卷积迭代t次
        for i in range(self.layer_num):
            # 把整个嵌入划分成factor_num个子嵌入表 每个factor上的维度是factor_dim
            ego_factor_emb_list=tf.split(all_embeddings[-1],self.factor_num,axis=1) # [U+I,factor_dim]
            # 每层卷积需要一个存放H_t的变量
            ego_factor_emb_list_t=[x for x in ego_factor_emb_list]

            # 【step 2】：动态路由 迭代n_iteration次
            for t in range(self.n_iteration):
                # 【step 2-1】：通过head_node和tail_node的内积 计算出在每个factor上的A_scores
                A_scores_list=[]
                # 计算所有边在每个factor上的score
                for k in range(self.factor_num):
                    # 取所有Edge的头节点
                    head_node_embeddings=tf.nn.embedding_lookup(ego_factor_emb_list_t[k],self.all_h_list) # [Edge,factor_dim]
                    # 取所有Edge的尾节点
                    tail_node_embeddings=tf.nn.embedding_lookup(ego_factor_emb_list[k],self.all_t_list) # [Edge,factor_dim]
                    # 计算score
                    A_scores_list.append(tf.reduce_sum(tf.multiply(head_node_embeddings,tail_node_embeddings),axis=1,keepdims=False)) # [Edge]

                # 获得A_scores_tensor 并softmax
                A_scores_tensor=tf.stack(A_scores_list,axis=0) # [Eage] -> [factor_num,Edge]
                A_scores_tensor=tf.nn.softmax(A_scores_tensor,axis=0) # [factor_num,Edge]
                #print(A_scores_tensor.get_shape)
                # [factor_num,Edge] -> [U+I,U+I] sp tensor * factor_num
                A_scores_sp_tensor_list=[self._create_sp_tensor(self.all_h_list,
                                                                self.all_t_list,
                                                                A_scores_tensor[i],
                                                                self.A_in_shape) for i in range(self.factor_num)]

                # 【step 2-2】：通过A_scores_sp_tensor在每个factor进行卷积操作
                for k in range(self.factor_num):
                    #print(ego_factor_emb_list_t[k].get_shape)
                    # 根据 H_0 和 S_t 计算 H_t+1 = H_0 + S_t * H_0
                    # [U+I,factor_dim] + [U+I,U+I] * [U+I,factor_dim]
                    ego_factor_emb_list_t[k]=ego_factor_emb_list[k]+tf.sparse.sparse_dense_matmul(A_scores_sp_tensor_list[k],ego_factor_emb_list[k])
                    # 将embedding限制在(-1,1)之间
                    ego_factor_emb_list_t[k]=tf.math.l2_normalize(ego_factor_emb_list_t[k], axis=1)                    

            # 【step 3】：完成一层的卷积操作
            all_embeddings.append(tf.concat(ego_factor_emb_list_t,axis=1))

        ## 把各层得到的嵌入相加再拆出user item emb
        #all_embeddings = tf.stack(all_embeddings, 1) # -> [U+I,layer_num+1,emb_dim]
        #all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False) # -> [U+I,emb_dim]
        
        # LightGCN、DGCF：将每一层的嵌入相加得卷积L层后的嵌入
        # GNUD：直接使用卷积的最后一层 但是每一层多了残差连接

        return all_embeddings[-1],A_scores_tensor


    # 获取每个factor空间上的嵌入表[投影得到]
    def _get_disen_embedding(self,emb_origin):
        factor_emb_list=[]
        for i in range(self.factor_num):
            factor_emb=tf.layers.dense(emb_origin,
                                        self.factor_dim,
                                        use_bias=True,
                                        kernel_regularizer=tf.keras.regularizers.l2(self.regs[2]),
                                        bias_regularizer=tf.keras.regularizers.l2(self.regs[2]),
                                        activation='relu',
                                        name="factor{}_layer".format(i),
                                        reuse=tf.AUTO_REUSE,
                                        ) # [N,factor_dim]
            factor_emb_list.append(factor_emb)

        return factor_emb_list

    # 构造独立性损失函数[class分类器]
    def _creat_factor_loss_class(self,factor_emb_list,ids):
        n_size=tf.dtypes.cast(tf.shape(ids)[0], tf.float32)
        emb_list=[]
        for emb in factor_emb_list:
            emb_list.append(tf.nn.embedding_lookup(emb,ids))
            
        # class factor:X
        X=tf.concat(emb_list,axis=0) # [bacth_size*factor_num,k]
        label_list=[]
        for i in range(self.factor_num):
            tmp=[0]*self.factor_num
            tmp[i]=1
            # [1,0,0] [0,1,0] [0,0,1]
            tmp=tf.cast(tmp,tf.float32) # (factor_num,)
            tmp=tf.expand_dims(tmp,0) # [1,facor_num]
            label_list.append(tf.tile(tmp,[n_size,1])) # [bacth_size,factor_num]

        # class factor:label
        label=tf.concat(label_list,axis=0) # [bacth_size*factor_num,k+factor_num]

        # TODO class factor:shuffle 应该在main中shuffle传到feed_dict中
        #X_label=tf.concat([X,label],axis=1) # [bacth_size*factor_num,factor_num]
        #X_label=tf.random_shuffle(X_label)
        #X,label=tf.split(X_label,num_or_size_splits=[self.factor_dim,self.factor_num],axis=1)

        # class nn
        for i in range(len(self.factor_class_layers)):
            layer_dim=self.factor_class_layers[i]
            X=tf.layers.dense(X,
                              layer_dim,
                              use_bias=True,
                              activation='relu',
                              name="factor_class_layer{}".format(i),
                              reuse=tf.AUTO_REUSE,
                              )

        X_logit=tf.layers.dense(X,
                                self.factor_num,
                                use_bias=True,
                                activation=None,
                                name="factor_class",
                                reuse=tf.AUTO_REUSE,
                                )

        #label = tf.tile(tf.eye(self.factor_num), [n_size, 1])

        # 计算softmax 交叉熵
        factor_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=X_logit))
        factor_loss=self.regs[1]*factor_loss

        return factor_loss

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

    # 构造成稀疏tensor
    def _create_sp_tensor(self,row,col,values,shape):
        indices = np.mat([row, col]).transpose() # [2,Edge] -> [Edge,2]
        # 创建稀疏矩阵、indices是index[row col]、data是值
        # 注：加上shape 因为可能有全0行 全0列 所以coo形式必须加shape
        return tf.SparseTensor(indices, values, shape)

    # 构造cf损失函数
    def _creat_cf_loss(self,u_e,pos_i_e,neg_i_e):

        pos_scores=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,1,K] [N,1,K] -> [N,1,K] -> [N,1]
        neg_scores=tf.reduce_sum(tf.multiply(u_e,neg_i_e),axis=1) # [N,1,K] [N,1,K] -> [N,1,K] -> [N,1]

        regular=tf.nn.l2_loss(u_e)+tf.nn.l2_loss(pos_i_e)+tf.nn.l2_loss(neg_i_e) # 1

        diff=tf.log(tf.nn.sigmoid(pos_scores-neg_scores)) # [N,1]

        mf_loss=-(tf.reduce_mean(diff)) # [N,1] -> 1
        reg_loss=self.regs[0]*regular

        return mf_loss,reg_loss

    # 构造独立性损失函数[cos距离]
    def _creat_factor_loss_cos(self,factor_emb_list,ids):
        emb_list=[]
        for emb in factor_emb_list:
            emb_list.append(tf.nn.embedding_lookup(emb,ids))
       
        def get_cos(X_list):
            sum_square=tf.square(tf.add_n(X_list)) # [N,k]
            square_sum=[]
            for X in X_list:
                square_sum.append(tf.square(X)) # [N,k]
            square_sum=tf.square(tf.add_n(square_sum))

            return 0.5*tf.sqrt(tf.reduce_sum(tf.square(sum_square-square_sum)))

        # 计算cos距离
        factor_loss=get_cos(emb_list)
        factor_loss=self.regs[1]*factor_loss

        return factor_loss

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
        return sess.run([self.opt,self.loss,self.mf_loss,self.factor_loss],feed_dict)

    # predict
    def predict(self,sess,feed_dict):
        batch_predictions=sess.run(self.batch_predictions,feed_dict)
        return batch_predictions

    # 
