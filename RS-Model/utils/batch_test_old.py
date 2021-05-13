'''
- train过程中的生成batch data 以及test集的表现
- 2021/5/13
'''

"""
# TODO

test
-1 model得到ratings

-2 使用topK 最大堆的方式获取r
   要计算每个用户的预测结果r[1,0,1,0,0,1] 这种形式 

-3 将r传入metric计算出每个用户的指标值

-4 多线程计算多个user的评价结果

"""

