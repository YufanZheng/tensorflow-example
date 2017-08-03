# -*- coding: utf-8 -*-

import tensorflow as tf

"""
    Tensorflow use different graphs to separate different calculations
"""

# GRAPH 1
g1 = tf.Graph()
with g1.as_default():
    # Init variable "v" in graph g1, and give value 0
    v = tf.get_variable(
            "v", initializer=tf.zeros_initializer(), shape=[1])

# GRAPH 2
g2 = tf.Graph()
with g2.as_default():
    # Init variable "v" in graph g2, and give value 1
    v = tf.get_variable(
            "v", initializer=tf.ones_initializer(), shape=[1])
    
# Read variable "v" in g1
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # Because we are in graph g1, the variable v should be 0
        print( sess.run( tf.get_variable("v") ) )
    
# Read variable "v" in g2
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # Because we are in graph g2, the variable v should be 1
        print( sess.run( tf.get_variable("v") ) )
        
"""
    Tensorflow default graph if constant is not within any Graph
"""

a = tf.constant( [1.0, 2.0], name="a" )

print( a.graph is tf.get_default_graph() )

"""
    Specify the GPU as services to run calculation
"""

g = tf.Graph()
# Specify the GPU for running
with g.device('/gpu:0'):
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    result = a + b