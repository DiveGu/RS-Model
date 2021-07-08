'''
- 1 train过程中的生成batch data
  2 计算test集的表现
- 2021/5/13
'''
import utils.metrics as metrics
from utils.parser import parse_args
from utils.load_data import *
import multiprocessing
import heapq
import numpy as np

# 导入parser
# 需要参数 评价指标的K值集合；数据集信息来构造load_data

cores = multiprocessing.cpu_count() // 2


args = parse_args()
Ks = eval(args.Ks)
data_path='{}experiment_data/{}/{}_{}/'.format(args.data_path,args.dataset,args.prepro,args.test_method)

data_generator = Data(data_path,args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

# 获取一条user在test上的表现 不计算auc
def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    """
    user_pos_test:user在test上真正的item ids
    test_items:真正评估的user要预测的item集合，即 [所有item]-[user在train中交互过的item]
    rating:user对于所有items的预测评分向量
    Ks:@K的集合
    """
    item_score = {}
    # iid-预测rating
    for i in test_items:
        item_score[i] = rating[i]

    # 选出user在test_items上预测评分最高的K个item id集合
    # k-v 按照v排序 返回对应的k
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    # 依次查看topK中的item id是否在ground truth （user_pos_test）中
    # 结果即为 r=[1,0,0,1,1,0]
    # 1表示这个位置的item id在ground truth中；0表示这个位置的item id不在ground truth中
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

# 计算单个用户的auc
def get_auc(item_score, user_pos_test):
    """
    item_score:单个user对[所有item-train item]的评分
    user_pos_test:user在测试集上的ground truth item id集合
    """
    # id-score 按照score降序排列
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

# 获取一条user在test上的表现 计算该用户的auc
def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

# 根据user对所有item的评分向量来计算评价指标
def get_performance(user_pos_test, r, auc, Ks):
    """
    user_pos_test:user在test上真正的item ids
    r:[1,0,0,1,1,0] 当前idx上的item in ground truth flag
    auc:
    Ks:@K集合
    """
    precision, recall, ndcg, hit_ratio = [], [], [], []

    # 计算项指标 注：传入user_pos_test是为了计算recall时用（做分母）
    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        # 当前uid在train中的历史items
        training_items = data_generator.train_user_dict[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    # 当前uid在test中的ground truth
    user_pos_test = data_generator.test_user_dict[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


# 测试模型的表现
def test(sess, model, users_to_test, drop_flag=False, batch_test_flag=False):
    """
    sess:sess
    model:模型对象
    users_to_test:要测试的users
    drop_flag:True-不需要考虑dropout
              False-需要考虑dropout 即feed_dict需要传入
    batch_test_flag:True-按照batch_size来预测user对item的评分
                    False-直接用所有的item id来预测user对所有item的评分
    """
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    # 多进程 多cpu并行
    # 当有新的请求提交到Pool中时，如果池还没有满，就会创建一个新的进程来执行请求
    # 如果池满，请求就会告知先等待，直到池中有进程结束，才会创建新的进程来执行这些请求
    pool = multiprocessing.Pool(cores)

    # 如果直接全量预测所有u对所有i的评分，内存可能不够
    # 所以先划分user batch；再根据batch_test_flag决定是否划分item batch
    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1 # 测试uid的batch_size

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        #user_batch=np.reshape(np.array(user_batch),(-1,1)) # [u,1]

        if batch_test_flag:

            n_item_batchs = ITEM_NUM // i_batch_size + 1
            # 每个user对所有item的预测得分向量 
            # 对于每个user来说 长度=ITEM_NUM
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)
                
                # 当前item batch的item的所有id集合
                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch})
                else:
                    i_rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                model.pos_items: item_batch,
                                                                model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                model.mess_dropout: [0.]*len(eval(args.layer_size))})
                # 更新当前batch的预测矩阵
                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM

        else:
            item_batch = range(ITEM_NUM)
            #item_batch=np.reshape(np.array(item_batch),(-1,1)) # [i,1]

            if drop_flag == False:
                rate_batch=model.predict(sess,{model.users: user_batch,
                                    model.pos_items: item_batch})
            else:
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                              model.pos_items: item_batch,
                                                              model.node_dropout: [0.] * len(eval(args.layer_size)),
                                                              model.mess_dropout: [0.] * len(eval(args.layer_size))})

        # 为当前的user_batch弄成了当前user_batch_size个元组，每个元组是(uid对所有item的评分向量,uid)
        user_batch_rating_uid = zip(rate_batch, user_batch)
        # pool多核cpu运行 当前user_batch_num个任务，对应的参数（即元组）zip
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        # batch_result是多核并行test_one_user()的结果集合
        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result



