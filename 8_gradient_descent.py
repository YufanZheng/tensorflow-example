# -*- coding: utf-8 -*-

"""
    Gradient Descent: 梯度下降
        - 反向的导数，乘以学习率
        - 沥青个问题：
            - 1，梯度下降容易产生局部最优解
            - 2，梯度下降是在所有的训练集上面进行的损失函数的计算，很费时
    Stochastic Gradient Descent：随机梯度下降
        - 随机采用一个数据集上面的损失函数来决定下降方向
        - 实际的运用之中，采用折中的方法，用一小部分数据集但不是全部
    Backpropagation: 反向传播 
        - 即是利用这些梯度下降方法来求矩阵的值
"""

import tensorflow as tf
from numpy.random import RandomState
import numpy as np

x = tf.placeholder( dtype=tf.float32, shape=[None, 2], name="x-input" )
y_ = tf.placeholder( dtype=tf.float32, shape=[None, 1], name="y-input" )

W = tf.Variable( tf.random_normal([2,1], stddev=1.0, seed=1) )
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x, W) + b

cross_entropy = -tf.reduce_mean( 
        y_ * tf.log( tf.clip_by_value(y, 1e-10, 1.0) ) )
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)

batch_size = 8

# Random datasets
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand( dataset_size, 2 )
# If x1 + x2 < 1 -> y = 1, else y = 0
Y = [ [int( x1 + x2 < 1 )] for (x1, x2) in X ]

with tf.Session() as sess:
    # Init
    tf.global_variables_initializer().run()
    STEPS = 5000
    
    start = np.random.randint(0, dataset_size - batch_size -1)
    end = start + batch_size
    
    # Training batch data
    for i in range(STEPS):
        sess.run( train_step, feed_dict = { x: X[start:end], y_:Y[start:end]})
        if i % 100 == 0:
            print( sess.run(W) )
    
    

