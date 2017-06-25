# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 22:27:37 2017

@author: Seagle
"""

#%%
import tensorflow  as tf

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
#%%  
y_ = tf.placeholder(tf.float32, [None, 2])

cImageSize = image_size
x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])

cl1Depth = 16
cl2Depth = 16
fcNoNodes = 512

W_conv1 = weight_variable([5, 5, 1, cl1Depth])
b_conv1 = bias_variable([cl1Depth])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
cImageSize = cImageSize//2

W_conv2 = weight_variable([5, 5, cl1Depth, cl2Depth])
b_conv2 = bias_variable([cl2Depth])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
cImageSize = cImageSize//2

W_fc1 = weight_variable([cImageSize * cImageSize * cl2Depth, fcNoNodes])
b_fc1 = bias_variable([fcNoNodes])

h_pool2_flat = tf.reshape(h_pool2, [-1, cImageSize*cImageSize*cl2Depth])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([fcNoNodes, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#%%
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
pred = tf.argmax(y_conv, 1)
correct_prediction = tf.equal(pred, tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()