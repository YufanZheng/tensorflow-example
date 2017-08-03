# -*- coding: utf-8 -*-

"""
    Neural Network Example
"""

import tensorflow as tf

from numpy.random import RandomState

# Define Batch Size
batch_size = 8

# Define Weights Matrix with random value
w1 = tf.Variable( tf.random_normal([2, 3], stddev=1, seed=1 ) )
w2 = tf.Variable( tf.random_normal([3, 1], stddev=1, seed=1 ) )

# Define placeholders
x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y-input')

# Defind neural network progagination
a = tf.matmul( x, w1 )
y = tf.matmul( a, w2 )

# loss function as cross entropy
cross_entropy = -tf.reduce_mean( 
        y_ * tf.log( tf.clip_by_value(y, 1e-10, 1.0) ) )

# Gradient descent to minimize
train_step = tf.train.GradientDescentOptimizer( 0.001 ).minimize( cross_entropy )

# Random datasets
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand( dataset_size, 2 )
# If x1 + x2 < 1 -> y = 1, else y = 0
Y = [ [int( x1 + x2 < 1 )] for (x1, x2) in X ]

# Session to run result
with tf.Session() as sess:
    
    # Init all variables
    tf.global_variables_initializer().run()
    
    # Have a look at the init Weights
    print( "Initial Weight Matrix 1 : \n", sess.run(w1) )
    print( "Initial Weight Matrix 2 : \n", sess.run(w2) )
    
    # Define how many steps in the training
    STEPS = 5000
    for i in range(STEPS):
        # Select a batch of dataset for training
        start = ( i * batch_size ) % dataset_size
        end = min(start+batch_size, dataset_size)
        
        # Feed the neural network with data
        sess.run( train_step, feed_dict = {
                x  : X[start:end],
                y_ : Y[start:end]
                } )
        
    
        # Print total entropy every 100 steps
        if i % 100 == 0:
            total_entropy = sess.run( cross_entropy, feed_dict = {
                x  : X, y_ : Y} )
            print( 'After {} steps, total cross entropy is {}'.format( i, total_entropy ) )
    
    # Have a look at the final Weights
    print( "Final Weight Matrix 1 : \n", sess.run(w1) )
    print( "Final Weight Matrix 2 : \n", sess.run(w2) )
        