# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 21:52:02 2017
Final Product File
@author: Seagle
"""
#%%
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 17:27:08 2017
The convoluntional one.
@author: Seagle
"""
#%%
from __future__ import print_function
from six.moves import cPickle as pickle
import tensorflow  as tf
import numpy as np

#%load PersonalDirectory.py

#%%
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
#%% Only need to reform the labels this time.
image_size = 64
num_labels = 2
num_channels = 1
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size,
                             num_channels)).astype(np.float32)
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
tf.test_dataset = tf.constant(test_dataset)

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

train_accuracy = 0.0

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(500):
    train_step.run({x: train_dataset, y_: train_labels, keep_prob: 0.5})
    if i % 100 == 0 and train_accuracy <0.99:
      train_accuracy = accuracy.eval(feed_dict={
          x: train_dataset, y_: train_labels, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    if i % 500 == 0:  
      test_accuracy = accuracy.eval(feed_dict={
          x: test_dataset, y_: test_labels, keep_prob: 1.0})
      print('step %d, test accuracy %g' % (i, test_accuracy)) 
  saver.save(sess, save_path = './SavedNN/model.ckpt')
  #pred = sess.run(pred, feed_dict = {x: test_dataset})
  
#%%
with tf.Session() as session:
    # restore the model
    saver.restore(session, "./SavedNN/model.ckpt")
    P = session.run(pred, feed_dict={x: test_dataset, keep_prob: 1.0})
  
#%%  
def comp_accuracy(pred, labels):
  return (100.0 * np.sum(pred == np.argmax(labels, 1))
          / pred.shape[0])
comp_accuracy(P, test_labels)



#%% OpenCV to take picture
import cv2
#from PIL import Image
# Camera 0 is the integrated web cam on my netbook
camera_port = 0
#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30
camera = cv2.VideoCapture(camera_port)

def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im
# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in range(ramp_frames):
 temp = get_image()
print("Taking image...")
# Take the actual image we want to keep
camera_capture = get_image()
cv2.imwrite("./SavedPict/temp.png", camera_capture)
del(camera)

#%% Open FR to find the face
import face_recognition as fr

face_locations = fr.face_locations(camera_capture)

from PIL import Image
top, right, bottom, left = face_locations[0]
face_image = camera_capture[top:bottom, left:right]
pil_image = Image.fromarray(face_image)
pil_image.show()

#%% Resize it into the TF size
import PIL
num_channels = 1
current_face = pil_image.resize((image_size,image_size), PIL.Image.ANTIALIAS)
current_face = current_face.convert('L')
current_face_org = pil_image.resize((image_size,image_size), PIL.Image.ANTIALIAS)
model_input = np.asarray(current_face, dtype = "int32").reshape((-1, image_size, image_size,
                             num_channels)).astype(np.float32)

#%%
saver = tf.train.Saver()
with tf.Session() as session:
    # restore the model
    saver.restore(session, "./SavedNN/model.ckpt")
    P = session.run(pred, feed_dict={x: model_input, keep_prob: 1.0})
#%% Does not work from here.
import os
class FaceMatcher(object):
    def __init__(self, dir):
        file_list = os.listdir(dir)
        self.factory_data = []
        self.filename = []
        for f in file_list:
            if f == "desktop.ini":
                continue
            img = Image.open(dir+'/'+f)
            self.factory_data.append(np.asarray(img, dtype = "int32"))
            self.filename.append(dir+'/'+f)

    def distance_to(self, from_image, to_image):
        distance = 0
        for idx in range(len(from_image)):
            distance = distance + sum([x - y for x, y in zip(from_image[idx], to_image[idx])])

        return distance / len(from_image)

    def closest_to(self, image):
        smallest_distance = self.distance_to(image, self.factory_data[0])
        closest_image = self.filename[0]
        for idx in range(1,len(self.factory_data)):
            current_distance = self.distance_to(image, self.factory_data[idx])
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                closest_image = self.filename[idx]

        return closest_image
#%%
gdpath = "C:/Users/Seagle/Google Drive/MHunt/DATA"
label_PathMap = {0: gdpath + "/Negative", 1: gdpath + "/Positive"}
match_emoji = FaceMatcher(label_PathMap[P[0]])

match_emoji.closest_to(np.asarray(current_face_org, dtype = "int32"))

distance = 0
aa = match_emoji.factory_data[0]
bb = match_emoji.factory_data[1]
for idx in range(len(aa)):
    distance = distance + sum([x - y for x, y in zip(aa[idx], bb[idx])])

        
#%% Output the emotion: positive negative
gdpath = "C:/Users/Seagle/Google Drive/MHunt/DATA/Negative"
label_PathMap = {0: gdpath + "/Negative", 1: gdpath + "/Positive"}

#match_emoji = FaceMatcher(label_PathMap[P[0]])
last_input = np.asarray(current_face_org).tolist()

img = Image.open(gdpath+'/'+"Angry_Devil_Emoji.jpg")
print (np.array(img).tolist())
print (last_input)
output_pict = match_emoji.closest_to(last_input)

#%% Match a image file.
 match_emoji = FaceMatcher(label_PathMap[P[0]])



#%%
import random
folder = gdpath + "/Positive"
image_files = os.listdir(folder)

image_sample = random.sample(image_files, 5)
for image in image_sample:
    image_file = folder + "/" + image
    i = PIL.Image.open(image_file)
