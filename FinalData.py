# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 00:29:23 2017
Reproduce the analyze data.
@author: Seagle
"""

#%%
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
#from IPython.display import display, Image
from six.moves import cPickle as pickle
import PIL
import random
image_size = 64
gdpath = "C:/Users/Seagle/Google Drive/MHunt"
#%%
def rImage2np_std(file):
    i = PIL.Image.open(file).convert("L")
    image_data = np.asarray(i, dtype = "int32")
    tMax = image_data.max()
    tMin = image_data.min()
    tRange = tMax - tMin
    image_data = (image_data - tMin)/tRange
    return image_data

def load_image(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = folder + "/" + image
    try:
      image_data = rImage2np_std(image_file)
      #image_data = (ndimage.imread(image_file).astype(float) - 
      #              pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
#%% 
folder0 = gdpath + "/DATA/Negative2"
nDatSet = load_image(folder0, 160)

folder1 = gdpath + "/DATA/Positive2"
pDatSet = load_image(folder1, 160)

#%% Decide the training and testing data set.
random.seed(42)
negTstInd = random.sample(range(nDatSet.shape[0]), 50)
posTstInd = random.sample(range(pDatSet.shape[0]), 50)
tstDat = np.concatenate((nDatSet[negTstInd, :, :], 
                          pDatSet[posTstInd, :, :]))
tstLabel = np.concatenate((np.zeros(50, dtype = "int32"),
                            np.ones(50, dtype = "int32")))
#%%
trnPD = np.delete(pDatSet, posTstInd, 0)
trnND = np.delete(nDatSet, negTstInd, 0)
trnDat = np.concatenate((trnND, trnPD))
trnLabel = np.concatenate((np.zeros(trnND.shape[0], dtype = "int32"),
                            np.ones(trnPD.shape[0], dtype = "int32")))

#%%
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(trnDat, trnLabel)
test_dataset, test_labels = randomize(tstDat, tstLabel)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', test_dataset.shape, test_labels.shape)

#%%
pickle_file = 'final.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise