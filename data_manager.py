# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class DataManager(object):
  def __init__(self, input_name):
    self.input_name = input_name
    
  def load(self):
    self.imgs = np.load('data/{0}/Data.npy'.format(self.input_name))

    self.n_samples = self.imgs.shape[0]
    self.n_channels = 1 if len(self.imgs.shape) == 3 else 3
    
  @property
  def sample_size(self):
    return self.n_samples

  def get_images(self, indices):
    images = []
    # print(indices, 'after')
    for index in indices:
      if self.n_channels == 1: # BW images
        img = self.imgs[index].reshape(64*64)
      elif self.n_channels == 3: # Colour images
        img = self.imgs[index].reshape(64*64*3)
      images.append(img)
    return images

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    # print(indices, 'before')
    return self.get_images(indices)
