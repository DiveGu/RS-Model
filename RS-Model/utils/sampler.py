import numpy as np
import scipy.sparse as sp

class Sampler(object):
    def __init__(self, user_num, item_num, num_ng=4, sample_method='item-desc', sample_ratio=0):
        """
        negative sampling class for some algorithms
        Parameters
        ----------
        user_num: int, the number of users
        item_num: int, the number of items
        num_ng : int, # of nagative sampling per sample
        sample_method : str, sampling method
                        'uniform' discrete uniform
                        'item-desc' descending item popularity, high popularity means high probability to choose
                        'item-ascd' ascending item popularity, low popularity means high probability to choose
        sample_ratio : float, scope [0, 1], it determines what extent the sample method except 'uniform' occupied
        """
        self.user_num = user_num
        self.item_num = item_num
        self.num_ng = num_ng
        self.sample_method = sample_method
        self.sample_ratio = sample_ratio

        assert sample_method in ['uniform', 'item-ascd', 'item-desc'], f'Invalid sampling method: {sample_method}'
        assert 0 <= sample_ratio <= 1, 'Invalid sample ratio value'

    # 为train取样 每条样本取多少/每个用户取多少
    def build_train_neg():
        return 0


    def transform(self, sampled_df, is_training=True):
        """

        Parameters
        ----------
        sampled_df : pd.DataFrame, dataframe waiting for sampling
        is_training : boolean, whether the procedure using this method is training part

        Returns
        -------
        neg_set : List, list of (user, item, rating, negative sampled items)
        """
        if not is_training:
            neg_set = []
            for _, row in sampled_df.iterrows():
                u = int(row['user'])
                i = int(row['item'])
                r = row['rating']
                js = []
                neg_set.append([u, i, r, js])
            
            return neg_set

        user_num = self.user_num
        item_num = self.item_num
        pair_pos = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for _, row in sampled_df.iterrows():
            pair_pos[int(row['user']), int(row['item'])] = 1.0

        neg_sample_pool = list(range(item_num))
        popularity_item_list = sampled_df['item'].value_counts().index.tolist()
        if self.sample_method == 'item-desc':
            neg_sample_pool = popularity_item_list
        elif self.sample_method == 'item-ascd':
            neg_sample_pool = popularity_item_list[::-1]
        
        neg_set = []
        uni_num = int(self.num_ng * (1 - self.sample_ratio))
        ex_num = self.num_ng - uni_num
        for _, row in sampled_df.iterrows():
            u = int(row['user'])
            i = int(row['item'])
            r = row['rating']

            js = []
            for _ in range(uni_num):
                j = np.random.randint(item_num)
                while (u, j) in pair_pos:
                    j = np.random.randint(item_num)
                js.append(j)
            for _ in range(ex_num):
                if self.sample_method in ['item-desc', 'item-ascd']:
                    idx = 0
                    j = int(neg_sample_pool[idx])
                    while (u, j) in pair_pos:
                        idx += 1
                        j = int(neg_sample_pool[idx])
                    js.append(j)
                else:
                    # maybe add other sample methods in future, uniform as default
                    j = np.random.randint(item_num)
                    while (u, j) in pair_pos:
                        j = np.random.randint(item_num)
                    js.append(j)
            neg_set.append([u, i, r, js])

        print(f'Finish negative samplings, sample number is {len(neg_set) * self.num_ng}......')

        return neg_set

    # 为test用户构造候选集 用于rank
    def build_candidates_set(test_ur, train_ur, item_pool, candidates_num=1000):
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
                samples = random.sample(v, candidates_num)
                test_ucands[k] = list(set(samples))
            else:
                samples = random.sample(sub_item_pool, sample_num) # 从候选item池中采样num个
                test_ucands[k] = list(v | set(samples)) # 将采样的items和已有的items合并
    
        return test_ucands