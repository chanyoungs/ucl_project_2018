# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class DataManager(object):
  def __init__(self, input_name):
    self.input_name = input_name
    
  def load(self):
    self.imgs = np.load(f'data/{self.input_name}/Data.npy')
    self.n_samples = self.imgs.shape[0]
    self.n_channels = 1 if len(self.imgs.shape) == 3 else 3
    self.input_size = np.prod(self.imgs[0].shape)
    
    self.imgs_dis = np.load(f'./data/{self.input_name}/Metric_data.npy')
    self.imgs_dis = self.imgs_dis.reshape(self.imgs_dis.shape[0], self.imgs_dis.shape[1], -1)
    self.latents_gt = np.load(f'./data/{self.input_name}/Metric_gt.npy')
    
  @property
  def sample_size(self):
    return self.n_samples

  def get_images(self, indices):
    return self.imgs[indices].reshape(len(indices), -1)
    # return [self.imgs[index].reshape(-1) for index in indices]
    # return [self.imgs[index].reshape(64*64*self.n_channels) for index in indices]

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    # print(indices, 'before')
    return self.get_images(indices)
