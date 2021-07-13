"""
- NeuMF
- 2021/7/13
"""
import tensorflow.compat.v1 as tf
import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class NeuMF():
    def __init__(self,data_config,pretrain_data,args):
        self.model_type='neumf'
        self.pretrain_data=pretrain_data

        self.n_users=data_config['n_users']
        self.n_items=data_config['n_items']

        self.verbose=args.verbose

        self.emb_dim=args.embed_size
        self.layers=eval(args.layers)
        self.lr=args.lr

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

        # 模型参数量
        self._static_params()

    # 模型搭建
    def _forward(self):
        pos_predict,pos_emb_list=self._model_forward(self.users,self.pos_items)
        neg_predict,neg_emb_list=self._model_forward(self.users,self.neg_items)

        # 预测评分
        self.batch_predictions=pos_predict

        # 构造损失函数 优化
        self.mf_loss,self.reg_loss=self._creat_cf_loss(pos_predict,neg_predict,pos_emb_list,neg_emb_list)
        self.loss=self.mf_loss+self.reg_loss

        self.opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # 模型搭建
    def _model_forward(self,u,i):

        # 查嵌入表获得表示
        #self.weights['user_embedding']=tf.math.l2_normalize(self.weights['user_embedding'], axis=1)
        #self.weights['item_embedding']=tf.math.l2_normalize(self.weights['item_embedding'], axis=1)

        u_e=tf.nn.embedding_lookup(self.weights['user_embedding'],u) # 
        i_e=tf.nn.embedding_lookup(self.weights['item_embedding'],i)
        
        mlp_u_e=tf.nn.embedding_lookup(self.weights['user_embedding'],u)
        mlp_i_e=tf.nn.embedding_lookup(self.weights['item_embedding'],i)
        
        # GMF
        gmf=tf.multiply(u_e,i_e)
        # MLP
        mlp_input=tf.concat([mlp_u_e,mlp_i_e],axis=1) # [N,k] [N,k] -> [N,2k]
        mlp_output=mlp_input
        for layer_size in self.layers:
            mlp_output=tf.layers.dense(mlp_output,layer_size,activation='relu')
        # GMF+MLP
        predict_input=tf.concat([gmf,mlp_output],axis=1)
        predict_ouput=tf.layers.dense(mlp_output,1) # [N,1]

        emb_list=[u_e,i_e,mlp_u_e,mlp_i_e]

        return predict_ouput,emb_list

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
    def _creat_cf_loss(self,pos_predict,neg_predict,pos_emb_list,neg_emb_list):
        for emb in pos_emb_list+neg_emb_list:
            regular=tf.nn.l2_loss(emb)# 1

        diff=tf.log(tf.nn.sigmoid(pos_predict-neg_predict)) # [N,1]

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
