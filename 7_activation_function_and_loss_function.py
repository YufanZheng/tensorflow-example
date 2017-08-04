#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:46:06 2017

@author: yufan
"""

import tensorflow as tf

"""
    Related Questions When build a neural network
    
    1. 线性模型的局限性：在神经网络中这个线性不可分的问题的解决是通过非线性的激活函数来进行的：
        在tensorflow中的激活函数有（常用的）：
            ReLU ：tf.nn.relu( tf.matmul(x, w) +b )
            Sigmoid ：tf.nn.sigmoid( tf.matmul(x, w) +b )
            tanh：tf.nn.tanh( tf.matmul(x, w) +b )
            
    2. 神经网络中的异或问题，比如x1 和 x2 一个为正一个为负，输出y为1 的这种情况， 感知机的模型是无法解决的
        为了解决这个问题，加入了隐藏层

    3. 损失函数，常用的损失函数：
        cross entropy： 交叉熵：指的是两个概率分部之间的距离
            如果概率分布的 0 - 1 之间，且总和为 1，则可以通过cross entropy来当做损失函数
            cross_entropy = -tf.reduce_mean( y_ * tf.log( tf.clip_by_value(y, 1e-10, 1.0 ) ) )
                其中 clip_by_value 函数是把 y 的值固定在 后面的范围内，放置log 0 和 log 1 的出现
            交叉熵一般是与softmax联合在一起用
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)

    4. 自定义损失函数
        loss = tf.reduce_sum( tf.where( tf.greater( v1, v2 ), (v1 - v2) * a, (v1 - v2) * b ) )
        tf.where - 第一个为condition，第二个和第三个是情况下做什么事情
"""

v1 = tf.constant( [1.0, 2.0, 3.0, 4.0] ) 
v2 = tf.constant( [4.0, 3.0, 2.0, 1.0] )

sess = tf.InteractiveSession()
print( tf.greater(v1, v2).eval() )

print( tf.where( tf.greater(v1, v2), v1, v2 ).eval() )
sess.close()

from numpy.random import RandomState

batch_size = 8

# Input and output
x = tf.placeholder( dtype=tf.float32, shape=[None, 2], name="x-input" )
y_ = tf.placeholder( dtype=tf.float32, shape=[None, 1], name="y-input" )

# Weights
w = tf.Variable( tf.random_normal( shape=[2,1], stddev=1, seed=1 ) )
y = tf.matmul( x, w )

# 自定义损失函数
loss_less = 10
loss_more = 1

loss = tf.reduce_sum( tf.where( tf.greater(y, y_),
                               (y - y_) * loss_more,
                               (y_ - y) * loss_less ) )

train_step = tf.train.AdamOptimizer(0.001).minimize( loss )

rdm = RandomState(1)

# Random dataset
dataset_size = 128
X = rdm.rand( dataset_size, 2 )
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in X]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    STEPS = 5000
    for i in range(STEPS):
        start = ( i * batch_size ) % dataset_size
        end = min( start + dataset_size, dataset_size )
        sess.run( train_step, 
                 feed_dict={ x: X, y_:Y } )
        if i % 100 == 0:
            print( sess.run(w) )














