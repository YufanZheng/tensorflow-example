#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:39:17 2017

@author: yufan
"""

import tensorflow as tf

"""
 Notes: 
    
 a, b, result are tensors
 Tensor is an high dimensional array but it doesn't save the data,
 It contains 3 information:
   1. name -> How this variable is calculated
   2. shape
   3. type
   
"""
a = tf.constant([1.0, 2.0], dtype=tf.float32, name="a")
b = tf.constant([2.0, 3.0], dtype=tf.float32, name="b")

result = a + b

#

with tf.Session() as sess:
    sess.run(result)
    print result.eval()
