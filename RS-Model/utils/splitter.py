import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, GroupShuffleSplit

# 划分数据集为train data和test data
def split_test(df, test_method='fo', test_size=.2):
    """
    - parameters
    df : pd.DataFrame raw data waiting for test set splitting
    test_method : str, 划分测试集的方式
                    'fo': split by ratio 单纯按比例划分
                    'tfo': split by ratio with timestamp 考虑时间带比例划分
                    'tloo': leave one out with timestamp 考虑时间的留一法
                    'loo': leave one out 留一法
                    'ufo': split by ratio in user level 按照每个用户进行划分
                    'utfo': time-aware split by ratio in user level 考虑时间按照每个用户进行划分
    test_size : float, 测试集比例

    - returns
    train_set : pd.DataFrame training dataset
    test_set : pd.DataFrame test dataset

    """
    train_set, test_set = pd.DataFrame(), pd.DataFrame()


    # 原始code：100个用户 20个用户的数据作为test 那这样划分，test的用户嵌入得不到训练啊？？
    if test_method == 'oldufo':
        driver_ids = df['user']
        # 原始df的uid已经重新编码了，为何还需要这一步操作??
        # 分别是：[升序排列取值set]；[重新编码id原始list] 
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=2020)
        # 按照用户组划分
        for train_idx, test_idx in gss.split(df, groups=driver_indices):
            train_set, test_set = df.loc[train_idx, :].copy(), df.loc[test_idx, :].copy()

    # 自己想法：每个用户交互历史随机比例作为test
    elif test_method=='ufo':
        # 可以加一个打乱reset_index 写成 utfo那样
        # 也可以写成loo那样
        # 对每一组：从index中随机抽取,不重复的test_size的index

        # 保证test_num>=1 不然可能返回空[] 后续的df.loc报错 或者后续检查test_index有无空值
        test_index=df.groupby(['user']).apply(\
            lambda grp: np.random.choice(grp.index,\
            max(int(np.floor(len(grp)*test_size)),1),\
            replace=False)).explode().values
        
        # 如果不保证test_num>=1，就得加下面这一行
        # test_index=list(filter(None, test_index))
        test_set=df.loc[test_index,:]
        train_set=df[~df.index.isin(test_index)]
        train_set=train_set.sort_values(['user']).reset_index(drop=True)
        test_set=test_set.sort_values(['user']).reset_index(drop=True)


    # 每个用户的交互记录中挑选最新比例的交互作为test
    elif test_method == 'utfo':
        df = df.sort_values(['user', 'timestamp']).reset_index(drop=True) # 按照时间升序排列
        def time_split(grp):
            # grp.index是当前user的所有交互的index
            start_idx = grp.index[0] # 第一个交互的index
            split_len = int(np.ceil(len(grp) * (1 - test_size))) # 向上取整 train数量
            split_idx = start_idx + split_len # test开始位置(start+train数量)
            end_idx = grp.index[-1] # test结束位置

            # 得到每一个用户的test iid列表 
            # 每一组df为 uid:[test iid集合]
            if(end_idx>split_idx):
                return list(range(split_idx, end_idx + 1))
            else:
                return [end_idx] # 如果test算出来个数为0 返回最后一个交互

        # apply 对groupby之后的每一个分组进行操作 返回series [uid,[test id集合]] 
        # user id:[test1,test2,test3]
        # explode() 某一列是[a,b,c] 转化成多行 [前面列,a] [前面列,b] [前面列,c]

        # df.groupby('user').apply(time_split) 返回series 
        # explode() 将 []列表分成单行
        # .vales将单行值转化成list
        test_index = df.groupby('user').apply(time_split).explode().values
        test_set = df.loc[test_index, :] # test_index是list
        train_set = df[~df.index.isin(test_index)]

    # 所有的交互数据 挑选最新比例
    elif test_method == 'tfo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(df) * (1 - test_size))) # train数据的index最后一个
        train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy() # 注意split_idx是取不到的(iloc取不到、loc取到)

    # 全部交互数据单纯按照比例划分
    elif test_method == 'fo':
        train_set, test_set = train_test_split(df, test_size=test_size, random_state=2019)

    # 每个用户留最新的1个交互 作为test
    elif test_method == 'tloo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False) # 降序排列 first表示取值相同取第一个
        train_set, test_set = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
        del train_set['rank_latest'], test_set['rank_latest']

    # 每个用户交互中随机挑一个 作为test
    elif test_method == 'loo':
        # # slow method
        # test_set = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        # test_key = test_set[['user', 'item']].copy()
        # train_set = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()

        # # quick method
        test_index = df.groupby(['user']).apply(lambda grp: np.random.choice(grp.index)).values
        test_set = df.loc[test_index, :].copy()
        train_set = df[~df.index.isin(test_index)].copy()


    train_set, test_set = train_set.reset_index(drop=True), test_set.reset_index(drop=True)
    print('train data num:{};test data num:{}'.format(train_set.shape[0],test_set.shape[0]))

    return train_set, test_set

# 划分train为train data和val data
def split_validation(train_set, val_method='fo', fold_num=1, val_size=.1):
    """
    - parameters
    train_set : pd.DataFrame train set waiting for split validation
    val_method : str, way to split validation
                    'cv': combine with fold_num => fold_num-CV
                    'fo': combine with fold_num & val_size => fold_num-Split by ratio(9:1)
                    'tfo': Split by ratio with timestamp, combine with val_size => 1-Split by ratio(9:1)
                    'tloo': Leave one out with timestamp => 1-Leave one out
                    'loo': combine with fold_num => fold_num-Leave one out
                    'ufo': split by ratio in user level with K-fold
                    'utfo': time-aware split by ratio in user level
    fold_num : int, the number of folder need to be validated, only work when val_method is 'cv', 'loo', or 'fo'
    val_size: float, the size of validation dataset

    - returns
    train_set_list : List, list of generated training datasets
    val_set_list : List, list of generated validation datasets
    cnt : cnt: int, the number of train-validation pair

    """
    if val_method in ['tloo', 'tfo', 'utfo']:
        cnt = 1
    elif val_method in ['cv', 'loo', 'fo', 'ufo']:
        cnt = fold_num
    else:
        raise ValueError('Invalid val_method value, expect: cv, loo, tloo, tfo')
    
    train_set_list, val_set_list = [], []
    if val_method == 'oldufo':
        driver_ids = train_set['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=fold_num, test_size=val_size, random_state=2020)
        for train_idx, val_idx in gss.split(train_set, groups=driver_indices):
            train_set_list.append(train_set.loc[train_idx, :])
            val_set_list.append(train_set.loc[val_idx, :])
    # 修改ufo
    elif val_method == 'ufo':
        for _ in range(fold_num):
            val_index=train_set.groupby(['user']).apply(\
                lambda grp:np.random.choice(grp.index,\
                np.floor(len(grp)*val_size),\
                replace=False)).explode().values

            val_set=train_set.loc[val_index,:].reset_index(drop=True).copy()
            sub_train_set=train_set[~train_set.index.isin(val_index)].reset_index(drop=True).copy()
            train_set_list.append(train_set)
            val_set_list.append(val_set)

    if val_method == 'utfo':
        train_set = train_set.sort_values(['user', 'timestamp']).reset_index(drop=True)
        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - val_size)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]

            return list(range(split_idx, end_idx + 1))
        val_index = train_set.groupby('user').apply(time_split).explode().values
        val_set = train_set.loc[val_index, :]
        train_set = train_set[~train_set.index.isin(val_index)]
        train_set_list.append(train_set)
        val_set_list.append(val_set)
    if val_method == 'cv':
        kf = KFold(n_splits=fold_num, shuffle=False, random_state=2019)
        for train_index, val_index in kf.split(train_set):
            train_set_list.append(train_set.loc[train_index, :])
            val_set_list.append(train_set.loc[val_index, :])
    if val_method == 'fo':
        for _ in range(fold_num):
            train, validation = train_test_split(train_set, test_size=val_size)
            train_set_list.append(train)
            val_set_list.append(validation)
    elif val_method == 'tfo':
        # train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(train_set) * (1 - val_size)))
        train_set_list.append(train_set.iloc[:split_idx, :])
        val_set_list.append(train_set.iloc[split_idx:, :])
    elif val_method == 'loo':
        for _ in range(fold_num):
            val_index = train_set.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
            val_set = train_set.loc[val_index, :].reset_index(drop=True).copy()
            sub_train_set = train_set[~train_set.index.isin(val_index)].reset_index(drop=True).copy()

            train_set_list.append(sub_train_set)
            val_set_list.append(val_set)
    elif val_method == 'tloo':
        # train_set = train_set.sample(frac=1)
        train_set = train_set.sort_values(['timestamp']).reset_index(drop=True)

        train_set['rank_latest'] = train_set.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        new_train_set = train_set[train_set['rank_latest'] > 1].copy()
        val_set = train_set[train_set['rank_latest'] == 1].copy()
        del new_train_set['rank_latest'], val_set['rank_latest']

        train_set_list.append(new_train_set)
        val_set_list.append(val_set)

    return train_set_list, val_set_list, cnt

