"""
- utils 数据预处理、评价指标
    - load_origin 加载数据 显式转隐式 filter/core处理
    - splitter 
      划分所有数据为train data 和test data 划分方式可选
      划分train数据为 train data和val data

    - data_preprocess 生成实验数据
    - load_data 加载处理好的数据、负采样
    - metrics 计算单个user的评价指标
    - batch_test 计算recall、hit等表现
    - parser 模型参数、实验参数的设置

- model 模型
    - BPRMF 
    - NCF
    - History Atten
    - LightGCN

- main 运行模型 保存结果

"""


"""
TODO
- 模型本身的保存 与 pretrain emb的保存
- main中的batch划分写到batch_test中
- 修改加载batch为tf.dataset
- 更快的计算recall@K
- 如何网格搜索选最佳参数


- 历史行为的模型 关键是输入的batch什么样子
- 图的模型 图稀疏矩阵

"""

