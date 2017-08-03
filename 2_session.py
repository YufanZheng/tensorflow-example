#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:39:17 2017

@author: yufan
"""

import tensorflow as tf

"""
 Notes on Tensors: 
    
 a, b, result are tensors
 Tensor is an high dimensional array but it doesn't save the data,
 It contains 3 information:
   1. name -> How this variable is calculated
   2. shape
   3. type
   
"""
a = tf.constant([1.0, 2.0],  name="a")
b = tf.constant([2.0, 3.0], dtype=tf.float32, name="b")

result = a + b

"""
    Create Session:
"""
with tf.Session() as sess:
    sess.run(result)
    
    # The following codes have the same functionalities
    print( sess.run(result) )
    print( result.eval() )
    
"""
    ConfigProto
        - allow_soft_placement : Allow CPU if not GPU
        - log_device_placement : Recore which node is placed at which machine for testing
"""
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)

with tf.Session(config=config) as sess:
    print( sess.run(result) )