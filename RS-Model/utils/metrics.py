"""
- 计算评价指标
- 2021/5/13
"""

import numpy as np
from sklearn.metrics import roc_auc_score

# r 是用户u预测结果Top列表 [1,0,0,1,0] 
# 1 代表这个位置的item在TOP预测中
# 0 代表这个位置的item不在TOP预测中


def recall_at_k(r,k,all_pos_num):
    r=np.array(r)[:k]
    return float(np.sum(r)/all_pos_num)


def precision_at_k(r,k):
    r=np.array(r)[:k]
    return float(np.sum(r)/len(r))

def hit_at_k(r,k):
    r=np.array(r)[:k]
    print(r.size)
    if(np.sum(r)>=1):
        return 1.
    else:
        return 0.

def dcg_at_k(r,k):
    r=np.array(r)[:k]
    if(r.size>0):
        return float(np.sum(r/np.log2(np.arange(2,r.size+2))))
    return 0.

def ndcg_at_k(r,k):
    #r=np.array(r)[:k]
    dcg_max = dcg_at_k(sorted(r, reverse=True),k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


"""
AP：PR线下面积 具体计算方式可以用
    -1 PASCAL VOC CHALLENGE，给定cut一组阈值 对于recall>阈值，得到max的precision，对precision取平均
    -2 所有不同的recall对应的点处的精度值做平均
https://www.zhihu.com/question/41540197?sort=created
https://blog.csdn.net/william_hehe/article/details/80006758

mAP：所有类别AP的平均值
"""
def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    # 获得[1,2,...,cut]不同K值下的precision 求平均
    # 个人理解：每增加一个K值就相当于得到一个Recall ；得到前min(cut,正例num)的所有Recall下的Precision
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs,cut):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r,cut) for r in rs])

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


#r=[1,1,0,1,0,0,0,1,0,1]

##result=recall_at_k(r,5,100)
##result=precision_at_k(r,5)
##result=hit_at_k(r,5)
##result=dcg_at_k(r,5)
#result=ndcg_at_k(r,5)

#print(result)
