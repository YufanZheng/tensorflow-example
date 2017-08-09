# -*- coding: utf-8 -*-

"""
    Script to train over the minst dataset
        - Use 2 hidden layers to train the dataset
        - Activation function : ReLU, Sigmoid
        - Gradient Descent
        - Dynamic giving learning rate
        - L2 regularization
        - Moving average model to make it more robust
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Get to understand the dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print( "Training data size: ", mnist.train.num_examples )
print( "Validation data size: ", mnist.validation.num_examples )
print( "Testing data size: ", mnist.test.num_examples )

print( "Example training data: ", mnist.test.images[0] )
print( "Example label", mnist.test.labels[0] )

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print( "X Shape: ", xs.shape )
print( "Y Shape: ", ys.shape )

# Use tensorflow to train the dataset

# MNIST dataset related constants
INPUT_NODE = 784    # Input nodes equals to the pixels of image
OUTPUT_NODE = 10    # Output nodes equals to the label 0-9 so it's 10

# Neural network related parameters
LAYER1_NODE = 500   # First hidden layer with 500 nodes
BATCH_SIZE = 100    # Batch Size
LEARNING_RATE_BASE = 0.8    # Base learning rate
LEARNING_RATE_DECAY = 0.99  # Learning rate decay
REGULARIZATION_RATE = 0.0001    # Reduce the model's complexity with regularization
TRAINING_STEPS = 30000          # Trainning steps
MOVING_AVERAGE_DECAY = 0.99     # Moving average decay rate

# Function to inference output from inputs
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu( tf.matmul(input_tensor, weights1) + biases1 )
        return tf.matmul( layer1, weights2 ) + biases2
    else:
        layer1 = tf.nn.relu( tf.matmul(input_tensor, avg_class.average(weights1) ) + avg_class.average(biases1) )
        return tf.matmul( layer1, avg_class.average(weights2) ) + avg_class.average(biases2)
    
# define the trainning process
def train(mnist):
    x = tf.placeholder( tf.float32, [None, INPUT_NODE], "x-input" )
    y = tf.placeholder( tf.float32, [None, OUTPUT_NODE], "y-input" )
    
    # Hidden layer parameters
    weights1 = tf.Variable( tf.truncated_normal(INPUT_NODE, LAYER1_NODE), stddev=0.1 )
    biases1 = tf.Variable( tf.constant(0.1, shape=[LAYER1_NODE]) )
    
    # Output layer parameters
    weights2 = tf.Variable( tf.truncated_normal(LAYER1_NODE, OUTPUT_NODE), stddev=0.1 )
    biases2 = tf.Variable( tf.constant(0.1, shape=[OUTPUT_NODE]) )
    
    # Do NOT use moving average
    y = inference( x, None, weights1, biases1, weights2, biases2 )
    
    # Make global step's trainable as False if don't need to compute the moving average
    global_step = tf.Variable(0, trainable=False)
    
    # Exponential Moving Average
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    
    # Use moving average
    average_y = inference( x, variable_averages, weights1, biases1, weights2, biases2 )
    
    # 