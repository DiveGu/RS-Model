"""
-DisenMF
- 2021/7/15
"""
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class DisenMF():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='DisenMF'
        self.pretrain_data=pretrain_data

        self.n_users=data_config['n_users']
        self.n_items=data_config['n_items']

        self.verbose=args.verbose

        # 各个factor上的嵌入是随机生成 OR 投影矩阵计算
        self.random_factor_emb=True
        self.factot_loss_type='class'

        self.emb_dim=args.embed_size
        self.factor_num=args.factor_num
        self.factor_dim=args.factor_dim
        self.lr=args.lr

        self.factor_activation=['relu','tanh','sigmoid']

        self.batch_size=args.batch_size
        self.regs=eval(args.regs)

        # 定义输入placeholder
        self.users=tf.placeholder(tf.int32,shape=[None,],name='users')
        self.pos_items=tf.placeholder(tf.int32,shape=[None,],name='pos_items')
        self.neg_items=tf.placeholder(tf.int32,shape=[None,],name='neg_items')

        # 初始化模型参数
        self.weights=self._init_weights()

        # 模型构造
        self._forward()

        ## 模型参数量
        #self._static_params()

    # 构造模型
    def _forward(self):
        # 1 把user item嵌入分别投影到factor的空间中
        if(self.random_factor_emb):
            user_factor_emb_list=self._get_disen_embedding_random('user')
            item_factor_emb_list=self._get_disen_embedding_random('item')
        else:
            user_factor_emb_list=self._get_disen_embedding(self.weights['user_embedding'])
            item_factor_emb_list=self._get_disen_embedding(self.weights['item_embedding'])

        # 2 获取计算score前的输入
        user_emb_final,item_emb_final=self._get_score_input(user_factor_emb_list,item_factor_emb_list)

        # 3 计算score_ui
        u_e=tf.nn.embedding_lookup(user_emb_final,self.users) # 
        pos_i_e=tf.nn.embedding_lookup(item_emb_final,self.pos_items)
        neg_i_e=tf.nn.embedding_lookup(item_emb_final,self.neg_items)

        # 预测评分
        self.batch_predictions=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,1]

        # 4 构造损失函数 优化
        self.mf_loss,self.reg_loss=self._creat_cf_loss(u_e,pos_i_e,neg_i_e)

        # 独立性损失
        if(self.factot_loss_type=='car'):
            self.factor_loss=self._creat_factor_loss_car(user_factor_emb_list,self.users)+\
                                self._creat_factor_loss_car(item_factor_emb_list,self.pos_items)+\
                                self._creat_factor_loss_car(item_factor_emb_list,self.neg_items)
        elif(self.factot_loss_type=='cos'):
            self.factor_loss=self._creat_factor_loss_cos(user_factor_emb_list,self.users)+\
                                self._creat_factor_loss_cos(item_factor_emb_list,self.pos_items)+\
                                self._creat_factor_loss_cos(item_factor_emb_list,self.neg_items)
        elif(self.factot_loss_type=='class'):
            self.factor_loss=self._creat_factor_loss_class(user_factor_emb_list,self.users)+\
            self._creat_factor_loss_class(item_factor_emb_list,self.pos_items)+\
            self._creat_factor_loss_class(item_factor_emb_list,self.neg_items)

        self.reg_loss+=self.factor_loss
        self.loss=self.mf_loss+self.reg_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        return 

    # 获取每个factor空间上的嵌入表[随机初始化得到]
    def _get_disen_embedding_random(self,name):
        factor_emb_list=[]
        for i in range(self.factor_num):
            factor_emb=self.weights['{}_embedding_{}'.format(name,i)]
            factor_emb_list.append(factor_emb)

        return factor_emb_list

    # 获取每个factor空间上的嵌入表[投影得到]
    def _get_disen_embedding(self,emb_origin):
        factor_emb_list=[]
        for i in range(self.factor_num):
            factor_emb=tf.layers.dense(emb_origin,
                                        self.factor_dim,
                                        use_bias=True,
                                        kernel_regularizer=tf.keras.regularizers.l2(self.regs[2]),
                                        bias_regularizer=tf.keras.regularizers.l2(self.regs[2]),
                                        activation=self.factor_activation[i%(len(self.factor_activation))],
                                        name="factor{}_layer".format(i),
                                        reuse=tf.AUTO_REUSE,
                                        ) # [N,factor_dim]
            factor_emb_list.append(factor_emb)

        return factor_emb_list

    # 根据user item在各个factor上的嵌入 获取最终计算score前的输入
    def _get_score_input(self,user_factor_emb_list,item_factor_emb_list):
        # 方法1：直接concat
        user_emb_final=tf.concat(user_factor_emb_list,axis=1) # [N,factor_num*factor_dim]
        item_emb_final=tf.concat(item_factor_emb_list,axis=1) # [N,factor_num*factor_dim]
        # 方法2：mlp得新的嵌入

        # 方法3：fwfm

        return user_emb_final,item_emb_final


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

        if(self.random_factor_emb):
            for i in range(self.factor_num):
                all_weights['user_embedding_{}'.format(i)]=tf.Variable(initializer([self.n_users,self.factor_dim]),name='user_embedding_{}'.format(i))
                all_weights['item_embedding_{}'.format(i)]=tf.Variable(initializer([self.n_items,self.factor_dim]),name='item_embedding_{}'.format(i))

        return all_weights

    # 构造cf损失函数
    def _creat_cf_loss(self,u_e,pos_i_e,neg_i_e):

        pos_scores=tf.reduce_sum(tf.multiply(u_e,pos_i_e),axis=1) # [N,K] [N,K] -> [N,K] -> (N,)
        neg_scores=tf.reduce_sum(tf.multiply(u_e,neg_i_e),axis=1) # [N,K] [N,K] -> [N,K] -> (N,)

        regular=tf.nn.l2_loss(u_e)+tf.nn.l2_loss(pos_i_e)+tf.nn.l2_loss(neg_i_e) # 1

        diff=tf.log(1e-12+tf.nn.sigmoid(pos_scores-neg_scores)) # [N,1]

        mf_loss=-(tf.reduce_mean(diff)) # [N,1] -> 1
        reg_loss=self.regs[0]*regular
       
        return mf_loss,reg_loss

    # 构造独立性损失函数[协方差距离]
    def _creat_factor_loss_car(self,factor_emb_list,ids):
        emb_list=[]
        for emb in factor_emb_list:
            emb_list.append(tf.nn.embedding_lookup(emb,ids))
        
        # 距离协方差 衡量X Y之间的非线性依赖关系
        # [n,k] [n,k] 由于每个样本才对应X和Y的依赖性 所以维度是n
        
        def get_distence(X):
            """
            X [n,k]
            A_ij = sqrt(|Xi|^2 + |Xj|^2 -2XiXj)
            A_ij = A_ij + mean(Ai) + mean(Aj)
            """
            A=tf.reduce_sum(tf.square(X),axis=1,keepdims=True) # [n,1]
            A=tf.sqrt(tf.maximum(A+tf.transpose(A)-2*tf.matmul(X,tf.transpose(X)),0.)+1e-8) # [n,1]+[1,n]+[n,n] -> [n,n]
            return A+tf.reduce_mean(A,axis=1,keepdims=True)+tf.reduce_mean(A,axis=0,keepdims=True)

        def get_ditence_cov(A,B):
            """
            Cov(A,B)=sqrt(SUM(AB)/n^2)
            """
            n=tf.dtypes.cast(tf.shape(A)[0], tf.float32)
            return tf.sqrt(tf.maximum(tf.reduce_sum(tf.multiply(A,B))/(n*n),0.)+1e-8)
            
        factor_loss=0.

        # 计算距离协方差
        for f1 in range(0,self.factor_num-1):
            for f2 in range(f1+1,self.factor_num):
                A=get_distence(emb_list[f1])
                B=get_distence(emb_list[f2])

                dcov_AA=get_ditence_cov(A,A)
                dcov_BB=get_ditence_cov(B,B)
                dcov_AB=get_ditence_cov(A,B)

                dcar_AB=dcov_AB/(tf.sqrt(tf.maximum(dcov_AA * dcov_BB, 0.0)) + 1e-9)
                
                factor_loss+=dcar_AB
                
        factor_loss/=(self.factor_num*(self.factor_num-1)/2)
        factor_loss=self.regs[1]*factor_loss

        return factor_loss

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

    
    # 构造独立性损失函数[class分类器]
    def _creat_factor_loss_class(self,factor_emb_list,ids):
        emb_list=[]
        for emb in factor_emb_list:
            emb_list.append(tf.nn.embedding_lookup(emb,ids))
            
        X=tf.concat(emb_list,axis=0) # [factor_num*N,k]
        X_logit=tf.layers.dense(X,
                                self.factor_num,
                                use_bias=True,
                                activation=None,
                                name="factor_class",
                                reuse=tf.AUTO_REUSE,
                                )
        n_size=tf.dtypes.cast(tf.shape(emb_list[0])[0], tf.float32)
        label = tf.tile(tf.eye(self.factor_num), [n_size, 1])

        # 计算softmax 交叉熵
        factor_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=X_logit))
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
