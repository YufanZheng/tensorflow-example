# -*- coding: utf-8 -*-

"""
    滑动平均模型：很多测试表明，采用了滑动平均模型之后的结果更好
        - 衰减率（decay）：用来控制模型训练的速度
        - 影子变量（shadow_variable）：
            - 初始值为 训练变量的值
            - 每一轮训练的时候，影子变量的值更新为：
                shadow_variable = decay * shadow_varaibel + ( 1 - decay ) * variable
                - shadow_variable : 影子变量的值
                - decay : 衰减率 = min { decay, (1+step)/(10+step) }
                - variable : 待更新的变量的值
"""

import tensorflow as tf

# 定义一个变量用于滑动平均，这个变量的初始值为0
# 这个变量的类型必须要为实数型
v1 = tf.Variable(0, dtype=tf.float32)
# 迭代的轮数，这里用来进行动态的控制衰减率
step = tf.Variable(0, trainable=False)

# ExponentialMovingAverage with decay 0.99 
ema = tf.train.ExponentialMovingAverage(0.99, step)
# Apply moving average operation on variable
maintain_average_op = ema.apply([v1])
# Get moving average value of a variable
obtain_average_op = ema.average(v1)

# 进行一下滑动平均的计算
with tf.Session() as sess:
    
    # Init all variable
    tf.global_variables_initializer().run()
    
    # First of all, have a llok at the initial moving average value 
    print( sess.run([v1, obtain_average_op]) )
    
    # Let's assign value 5 to v1 and see its moving average value
    sess.run( tf.assign(v1, 5) )
    sess.run( maintain_average_op )
    # v1 = 5
    # average_value = min{0.99, 0.1} * 0 + ( 1 - 0.1 ) * 5 = 4.5
    print( sess.run([v1, obtain_average_op]) )
    
    # Let's change v1 to 10, step to 10000
    sess.run( tf.assign(v1, 10) )
    sess.run( tf.assign(step, 10000) )
    sess.run( maintain_average_op )
    print( sess.run([v1, obtain_average_op]) )
    
    sess.run( maintain_average_op )
    print( sess.run([v1, obtain_average_op]) )