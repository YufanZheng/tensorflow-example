# -*- coding: utf-8 -*-

"""
    - 学习率的设置：
        - 学习率设置的太大，收敛快，但是不容易找到最小值（跨度太大）
        - 学习率太小，虽然可以找到精确的最小值，但是收敛太慢
        - 解决方案 ：
            - 动态的规划学习率：指数衰减法：
                - decayed_learning_rate = learning_rate * decayed_rate ^
                    ( globale_step / decayed_step )
                - decayed_learning_rate ： 每一轮优化的学习率
                - learning_rate ： 初始的学习率
                - decayed_rate ： 衰减系数
                - globale_step ： 当前的优化的论数
                - decayed_step ： 衰减速度
            - 在tensorflow中利用 tf.train.exponential_decay 来实现指数衰减
                - 除了上述的四个参数外，还有一个参数叫 staircase
                    - staircase: Boolean.  If `True` decay the learning rate at discrete intervals
                    - 意思是说，如果是设置为 True 的话，会成阶梯式的衰减学习率
"""

import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

"""
    - 动态的衰减学习率
"""
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1.0, global_step, 100, 0.96, staircase=True )
learning_step = tf.train.GradientDescentOptimizer( learning_rate ).minimize( loss=loss, global_step=global_step)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(learning_step, {x:x_train, y:y_train}) 
    if i % 10 == 0:
        print( 'Step :', i, sess.run( learning_rate, {x:x_train, y:y_train} )  )