import gc
import random
import numpy as np
import pandas as pd

from collections import defaultdict


# 加载数据 转化成user-item-rating 隐式行为
def load_rate(data_path,src='lastfm',prepro='origin',binary=True, pos_threshold=None, level='ui'):
    """
    - parameters
    src:数据集的名字
    prepro:str, 'origin', f'{N}-core', f'{N}-filter',预处理的方式
    binary:bool,转为CTR或者回归问题
    pos_threshold:int,显式行为转化成隐式行为的阈值
    level：str,'u','i','ui'，对user/item 预处理
    - return
    df:pd.df,user-item-rating-timestap
    uid_2_origin:dict，{新编码uid：原始uid}
    iid_2_origin:dict，{新编码iid：原始iid}
    """
    df=pd.DataFrame()
    # 读取原始数据 转化成df
    if src=='lastfm':
        # user_artists.dat
        df = pd.read_csv(data_path+'Last_FM/user_artists.dat', sep='\t')
        df.rename(columns={'userID': 'user', 'artistID': 'item', 'weight': 'rating'}, inplace=True)
        # fake timestamp column
        df['timestamp'] = df['item']

    # rating >= threshold 作为正样本
    if pos_threshold is not None:
        print('origin interaction num:{}'.format(df.shape[0]))
        df = df[df['rating']>=pos_threshold].reset_index(drop=True)
        print('after pos_threshold {}, pos interaction num:{}'.format(pos_threshold,df.shape[0]))

    # 隐式行为 pos label=1.0
    if binary:
        df['rating'] = 1.0

    # 数据预处理方式
    if prepro == 'origin':
        pass

    # fiter预处理
    elif prepro.endswith('filter'):
        # pattern = re.compile(r'\d+') # 正则表达式 d匹配[0-9]数字 +匹配连续多个数字
        # filter_num = int(pattern.findall(prepro)[0])
        filter_num = int(prepro.split('-')[0])
        tmp1 = df.groupby(['user'], as_index=False)['item'].count() # 不加as_index=False，默认uid为index，不是col
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        # 原有的df上加了两列：cnt_item,cnt_user
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])

        if level == 'ui':    
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()        

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()
    
    # core预处理
    elif prepro.endswith('core'):
        core_num = int(prepro.split('-')[0])

        # 按照每个用户的交互item数量，进行一次core_num的过滤
        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        # 按照每个物品的被交互user数量，进行一次core_num的过滤
        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                # chk_i长度相当于user数量，chk_u长度相当于item数量
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)

    print('after {} pre-data,interaction num:{}'.format(prepro,df.shape[0]))
    print(f'Finish loading [{src}]-[{prepro}] dataset')

    # 对原始uid重新编码 [0,1,2,.....user_num-1] 按顺序排列 用index代替原始uid
    # .codes为index，.categories为值
    user_codes=pd.Categorical(df['user'])
    item_codes=pd.Categorical(df['item'])

    user_num = df['user'].nunique()
    item_num = df['item'].nunique()

    # 字典 新uid:原始uid
    uid_2_origin=dict(zip(list(range(0,user_num)),df['user'].unique()))
    iid_2_origin=dict(zip(list(range(0,item_num)),df['item'].unique()))
    
    df['user'] = user_codes.codes # 替换df[user]列,将原始的uid换成新编码的uid
    df['item'] = item_codes.codes # 替换df[item]列,将原始的iid换成新编码的iid

    return df, uid_2_origin, iid_2_origin


# 获取user-item交互对
def get_ur(df):
    """
    - parameters
    df : pd.DataFrame, rating dataframe
    - return
    ur : dict, dictionary stored user-items interactions
    """
    ur = defaultdict(set) # 默认value为set类型
    for _, row in df.iterrows():
        ur[int(row['user'])].add(int(row['item']))

    return ur


# 获取item-user交互对
def get_ir(df):
    """
    - parameters
    df : pd.DataFrame, rating dataframe
    - return
    ir : dict, dictionary stored item-users interactions
    """
    ir = defaultdict(set) # 默认value为set类型
    for _, row in df.iterrows():
        ir[int(row['item'])].add(int(row['user']))

    return ir

# 为test用户构造候选集 用于rank
def build_candidates_set(test_ur, train_ur, item_pool, candidates_num=100):
    """
    - parameters
    test_ur : dict, ground_truth that represents the relationship of user and item in the test set
    train_ur : dict, this represents the relationship of user and item in the train set
    item_pool : the set of all items
    candidates_num : int, the number of candidates
    - returns
    test_ucands : dict, dictionary storing candidates for each user in test set
    """
    test_ucands = defaultdict(list)
    # 对于test的每个用户 进行采样
    for k, v in test_ur.items():
        sample_num = candidates_num - len(v) if len(v) < candidates_num else 0 # 取样数量=候选集数量-已有物品数量
        sub_item_pool = item_pool - v - train_ur[k] # 移去test中groud truth和train中交互的 item
        sample_num = min(len(sub_item_pool), sample_num)
        if sample_num == 0:
            #samples = random.sample(v, candidates_num)
            #test_ucands[k] = list(set(samples))
            test_ucands[k]=[] # 空
        else:
            samples = random.sample(sub_item_pool, sample_num) # 从候选item池中采样num个
            #test_ucands[k] = list(v | set(samples)) # 将采样的items和已有的items合并
            test_ucands[k]=list(samples)

    return test_ucands


