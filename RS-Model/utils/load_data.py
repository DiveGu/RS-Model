'''
加载处理好的数据
生成batch样本
'''
import json
import pandas as pd
import random as rd

class Data(object):
    def __init__(self, path, batch_size):

        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.csv'
        test_file = path + '/test.csv'

        # 用户数量 物品数量 train数量
        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0

        self.train_df, self.train_user_dict = self._load_ratings_train(train_file)
        self.test_df, self.test_user_dict = self._load_ratings_test(test_file)
        # train中的user集合
        self.exist_users = self.train_user_dict.keys()

        # 加载测试集的负采样
        #f = open(path+'test_neg.txt','r')
        #test_neg_dict=json.loads(f.read())

        self._statistic_ratings()

    # 加载训练集
    def _load_ratings_train(self,train_file):
        train_df=pd.read_csv(train_file)
        self.n_train=train_df.shape[0]
        user_dict=dict()
        ui_group = train_df.groupby(['user'], as_index=False)
        for i,j in ui_group:
            user_dict[i]=list(j['item'])
            
        self.n_users=train_df['user'].max()
        self.n_items=train_df['item'].max()
        return train_df,user_dict

    # 加载测试集
    def _load_ratings_test(self,test_file):
        test_df=pd.read_csv(test_file)
        self.n_test=test_df.shape[0]
        user_dict=dict()
        ui_group = test_df.groupby(['user'], as_index=False)
        for i,j in ui_group:
            user_dict[i]=list(j['item'])
            
        self.n_users=max(self.n_users,test_df['user'].max())
        self.n_items=max(self.n_items,test_df['item'].max())
        self.n_users+=1 # 数量要考虑idx==0 所以+1
        self.n_items+=1 # 数量要考虑idx==0 所以+1
        return test_df,user_dict


    # 输出数据的基本信息
    def _statistic_ratings(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    # 生成样本之前先为每一个正样本采样一个负样本 形成新的train_df [uid,pos_iid,neg_iid]
    def _sample(self):
        neg_list=[]
        for k,v in self.train_user_dict.items():
            # 为每个用户采样len(v) 个负样本
            sample_num=len(v)
            sub_item_pool=set(range(self.n_items))-set(self.train_user_dict[k])-set(self.test_user_dict[k])
            #print('用户id{}，交互数量{}，候选池数量{}'.format(k,len(set(self.train_user_dict[k])|set(self.test_user_dict[k])),len(sub_item_pool)))
            sample_num=min(len(v),len(sub_item_pool))
            if(sample_num==len(v)):
                negs=rd.sample(sub_item_pool, sample_num)
            else:
                # 当用户k的交互数量非常多的时候 item_pool的数量小于len(v) 反复采样 必须采样len(v)个
                negs=[]
                while(len(negs)<len(v)):
                    tmp_num=len(sub_item_pool) if len(v)-len(negs)>=len(sub_item_pool) else len(v)-len(negs)
                    negs=negs+rd.sample(sub_item_pool, tmp_num)
            neg_list=neg_list+negs

        return neg_list


    # 生成训练batch
    def generate_train_cf_batch(self,idx):
        #batch_num=self.train_df//self.batch_size
        if(idx==0):
            # 1 负采样
            self.df_copy=self.train_df[['user','item']]
            self.df_copy['neg_item']=self._sample()
            # 2 打乱数据
            self.df_copy=self.df_copy.sample(frac=1.0)
            #self.df_copy.to_csv(self.path+'sample.csv',index=False)

        # 3 生成batch数据
        start=idx*self.batch_size
        end=(idx+1)*self.batch_size
        end=end if end<self.n_train else self.n_train
        batch_data={
            'user':self.df_copy['user'][start:end].values,
            'pos_item':self.df_copy['item'][start:end].values,
            'neg_item':self.df_copy['neg_item'][start:end].values,
        }

        #return self.df_copy[start:end].values
        return batch_data 

    # 生成batch的feed_dict字典
    def generate_train_feed_dict(self,model,batch_data):
        feed_dict={
            model.users:batch_data['user'],
            model.pos_items:batch_data['pos_item'],
            model.neg_items:batch_data['neg_item']
        }
        
        return feed_dict

    # 根据uid_list和iid_list生成batch_data [进行预测score_ui时用；还有获取评价指标时使用]
    def generate_predict_cf_batch(self,uid_list,iid_list):
        batch_data={
            'user':uid_list,
            'pos_item':iid_list,
        }
        return batch_data

    # 生成预测batch的feed_dict字典
    def generate_predict_feed_dict(self,model,batch_data):
        feed_dict={
            model.users:batch_data['user'],
            model.pos_items:batch_data['pos_item'],
        }
        
        return feed_dict
