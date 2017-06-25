# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:52:21 2017

@author: Seagle
"""
#%%
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
#from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tensorflow  as tf
%load PersonalDirectory.py

#%% load data.
pickle_file = 'trial1.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

#%%
image_size = 64
num_labels = 2
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  # labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  new_labels = np.ndarray(shape = (labels.shape[0], 2), dtype = np.float32)
  new_labels[:, 0] = 1.0 - labels
  new_labels[:, 1] = labels
  return dataset, new_labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)  
  

#%%
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, image_size * image_size])
W = tf.Variable(tf.zeros([image_size * image_size, num_labels]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
#beta_regul = tf.constant(0.001)
tf.test_dataset = tf.constant(test_dataset)

sess.run(tf.global_variables_initializer())

#%%
beta_regul = tf.placeholder(tf.float32)
y = tf.matmul(x,W) + b
loss_function = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) 

#+\                               beta_regul * tf.nn.l2_loss(W)

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss_function)

#%%
for step in range(1000):
    sess.run(train_step, feed_dict={x: train_dataset, y_: train_labels})
    if (step % 500 == 0):
        print(step)
        print(sess.run(loss_function, {x : train_dataset, y_ : train_labels}))
#%%
test_prediction = tf.nn.softmax(tf.matmul(tf.test_dataset, W) + b)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
accuracy(test_prediction.eval(), test_labels) 