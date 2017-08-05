# -*- coding: utf-8 -*-

"""
    过拟合问题的解决方案一般是通过正则化的方法：
        - 正则化通过在原有的损失函数上加上一个 复杂度 函数
        - 复杂度 函数是由 权重 w 和 偏量 b 来决定的
        - 一阶正则化：
            - 复杂函数为权重的一阶范式：sum || w ||
            - 一阶正则化容易导致导致 权重的分布过于稀疏：
                - 即是很多 权重 都会 变为 0
                - 这是因为为了最小化损失函数，当权重为零的时候，复杂度损失函数为0
        - 二姐正则化：
            - 复杂函数的权重的二阶范式 sum ：|| w ^ 2 ||
            - 二阶正则化不会有使得权重过于稀疏的问题
                - 原因是当权重已经变为0.001 的时候，它的乘方已经就变成几乎为0了，
                    对损失函数已经没有太大的影响了
"""

# 一个简单的例子： 单层神经网络，在TensorFlow 中利用 正则化 来定义损失函数

import tensorflow as tf

x = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="x-input")
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y-input")

W = tf.Variable( tf.random_normal([2,1], stddev=1.0, seed=1) )

y = tf.matmul(x, W)

"""
    using L2 regularizer with lambda = 0.5 here
"""
loss = tf.reduce_mean( tf.square(y_ - y ) ) + tf.contrib.layers.l2_regularizer(0.5)(W)

# 一个完整的例子，利用循环来生成一个五层的神经网络

batch_size = 8
scale = 0.001

# 定义每一层网络中节点的个数
layer_dimension = [ 2, 10, 10, 10, 1 ]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深的节点，开始的时候是输入层
cur_layer = x
# 当前层的节点的个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成五层全连接的神经网络结构
for i in range(1, n_layers):
    # layer_dimension[i] 为下一层的节点数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = tf.Variable(tf.random_normal( [in_dimension, out_dimension] ), dtype=tf.float32)
    bias = tf.Variable( tf.constant(0.1, shape=[out_dimension]) )
    # add_to_collection 函数将这个新生成的变量的L2正则化的损失项加入集合
    tf.add_to_collection( "regularizer_losses", tf.contrib.layers.l2_regularizer(scale)(weight) )
    # 使用ReLu激活函数
    cur_layer = tf.nn.relu( tf.matmul(cur_layer, weight) + bias )
    # 进入下一层的节点数为党曾的节点数
    in_dimension = layer_dimension[i]

# 当前的层即为最终的输出层
y = cur_layer
# 损失函数的第一部分为方差的平均，第二部分为二阶正则化因子
loss = tf.reduce_mean( tf.square(y_ - y) ) + tf.add_n( tf.get_collection("regularizer_losses") )






