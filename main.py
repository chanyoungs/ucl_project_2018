# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from scipy.misc import imsave

import sys
input_name = sys.argv[1]
model_name = sys.argv[2]
channels = 3 # Colour vs Greyscale

path = os.path.join('outputs', model_name)

if not os.path.isdir(path):
    print('Path does not exist. Creating path...')
    os.makedirs(
        os.path.join(path, 'logs'))
    os.makedirs(
        os.path.join(path, 'figs_data'))
    os.makedirs(
        os.path.join(path, 'checkpoints'))
    os.makedirs(
        os.path.join(path, 'figures', 'Graphs'))
    os.makedirs(
        os.path.join(path, 'figures', 'Disentanglements'))
    os.makedirs(
        os.path.join(path, 'figures', 'Reconstructions'))
    print('Path created')
else:
    print('Path already exists')


from model import VAE
from data_manager import DataManager
# from data_manager_multiple import DataManager

tf.app.flags.DEFINE_integer("epoch_size", 20000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("gamma", 100.0, "gamma param for latent loss")
tf.app.flags.DEFINE_float("capacity_limit", 20.0,
                          "encoding capacity limit param for latent loss")
tf.app.flags.DEFINE_integer("capacity_change_duration", 100000,
                            "encoding capacity change duration")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "outputs/{0}/checkpoints".format(model_name), "checkpoint directory")
# tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./outputs/{0}/log", "log file directory".format(model_name))
# tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_boolean("training", True, "training or not")

flags = tf.app.flags.FLAGS

def train(sess, model, manager, saver):

    summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)
    
    n_samples = manager.sample_size
    
    reconstruct_check_images = manager.get_random_images(10)
    
    indices = list(range(n_samples))
    
    if not os.path.isfile(os.path.join(path, 'figs_data', 'figs_data.npy')):
        figs_data = {
            'epoch' : 0,
            'step' : 0,
            'latents' : [],
            'losses_r' : [], # Reconstruction losses over epochs for plot
            'losses_l' : [], # Latent losses over epochs for plot
            'disentangled_metric' : []
        }
    else:
        figs_data = np.load(os.path.join(path, 'figs_data', 'figs_data.npy'))
    
    # Training cycle
    while figs_data['epoch'] < flags.epoch_size:
        # Shuffle image indices
        random.shuffle(indices)
        
        total_batch = n_samples // flags.batch_size
        
        # Loop over all batches
        for i in range(total_batch):
            # Generate image batch
            batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
            batch_xs = manager.get_images(batch_indices)
            
            # Fit training using batch data
            reconstr_loss, latent_loss, summary_str = model.partial_fit(sess, batch_xs, figs_data['step'])
            
            figs_data['step'] += 1
            
        summary_writer.add_summary(summary_str, figs_data['epoch'])
        
        figs_data['losses_r'].append(reconstr_loss)
        figs_data['losses_l'].append(latent_loss)
        
        # Image reconstruction check & disentanglement check
        figs_data = plot_figures(sess, model, reconstruct_check_images, manager, figs_data)
        
        print("Epoch: {0} Loss_R: {1} Loss_L: {2}".format(figs_data['epoch'], reconstr_loss, latent_loss))
        
        # Save checkpoint
        saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = figs_data['epoch'])

        np.save(os.path.join(path, 'figs_data', 'figs_data.npy'), figs_data)

        figs_data['epoch'] += 1

def plot_figures(sess, model, images, manager, figs_data):
    epoch = figs_data['epoch']
    
    #################### Check image reconstruction
    x_reconstruct = model.reconstruct(sess, images)
    
    fig = plt.figure(facecolor='grey', figsize=(10, 4))
    for i in range(len(images)):
        image_shape = (64, 64) if images[0].shape[0] == 64*64 else (64, 64, 3)
        
        org_img = images[i].reshape(image_shape)
        org_img = org_img.astype(np.float32)
        
        reconstr_img = x_reconstruct[i].reshape(image_shape)
        
        plt.subplot(2, 10, i+1)
        plt.imshow(org_img)
        plt.axis('off')
        
        plt.subplot(2, 10, i+11)
        plt.imshow(reconstr_img)
        plt.axis('off')
        
    fig.suptitle('Original images(above), Reconstruction images(below) Epoch: {0}'.format(epoch))
    fig.savefig('./outputs/{0}/figures/Reconstructions/reconstruction{1}'.format(model_name, epoch))
    plt.close(fig) 
    
    ###################### Latent traversal
    img = manager.imgs[0].reshape(np.prod(image_shape))
    
    batch_xs = [img]
    z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
    z_sigma_sq = np.exp(z_log_sigma_sq)[0]
    
    figs_data['latents'].append(z_sigma_sq.reshape(10, 1))
    latent_graph = np.concatenate(figs_data['latents'], axis=1)
    
    # Save disentangled images
    z_m = z_mean[0]
    n_z = 10
    
    ind = np.argsort(np.array(figs_data['latents'][-1]).flatten())
    ind_inv = np.argsort(ind)
  
    fig = plt.figure(figsize=(10,10), facecolor='grey')
    for target_z_index in range(n_z):
        for ri in range(n_z):
            value = -3.0 + (6.0 / 9.0) * ri
            z_mean2 = np.zeros((1, n_z))
            for i in range(n_z):
                if( i == target_z_index ):
                    z_mean2[0][i] = value
                else:
                    z_mean2[0][i] = z_m[i]
            reconstr_img = model.generate(sess, z_mean2)
            rimg = reconstr_img[0].reshape(image_shape)
            
            plt.subplot(n_z, n_z, ind_inv[target_z_index]*n_z + ri+1)
            plt.imshow(rimg)
            plt.axis('off')
      
    fig.suptitle('Latent space traversal Epoch: {0}'.format(epoch))
    fig.savefig('./outputs/{0}/figures/Disentanglements/latent_space{1}'.format(model_name, epoch))
    plt.close(fig)
  
    ############### Graphs

    # Latent space
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(221)
    for v in range(n_z):
        plt.plot(range(latent_graph.shape[1]), latent_graph[ind[v],:], label=v+1)
    plt.legend()
    plt.title('Disentanglement over latent variables Epoch: {0}'.format(epoch))
    
    # Reconstruction loss
    plt.subplot(222)
    plt.plot(range(len(figs_data['losses_r'])), figs_data['losses_r'])
    plt.title('Reconstruction loss Epoch: {0}'.format(epoch))
    
    # Latent loss
    plt.subplot(223)
    plt.plot(range(len(figs_data['losses_r'])), figs_data['losses_r'])
    plt.title('Latent loss Epoch: {0}'.format(epoch))
    
    # Disentanglement metric
    images_total = manager.imgs[:10000]
    images_total = images_total.reshape(images_total.shape[0], -1)
    batch_xs = images_total
    z_mean_total, z_log_sigma_sq = model.transform(sess, batch_xs)
    z_mean_total_var = np.var(z_mean_total, axis=0)
    no_latents = 8
    
    print('Loading metric data...')
    images_dis = np.load('./data/{0}/disentanglement_metric_data.npy'.format(input_name))
    print('Data loaded!')
    images_dis = images_dis.reshape(images_dis.shape[0], images_dis.shape[1], -1)
    
    votes = np.zeros((10, no_latents))
    print('Counting votes...')
    for n in range(images_dis.shape[0]):
        batch_xs = images_dis[n]
        z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
        normalised = np.divide(z_mean, z_mean_total_var)
        normalised_var = np.var(normalised, axis=0)
        argmin = np.argmin(normalised_var)
        votes[argmin, n % no_latents] += 1
        
    print('Calculating accuracy...')
    accuracy = 0
    for n in range(10):
        accuracy += np.max(votes[n])
        
    accuracy *= 100 / images_dis.shape[0]
    print('Accuracy: {0}%'.format(accuracy))
    
    figs_data['disentangled_metric'].append(accuracy)
    
    plt.subplot(224)
    plt.plot(range(len(figs_data['disentangled_metric'])), figs_data['disentangled_metric'])
    plt.title('Disentanglement Metric Epoch: {0}'.format(epoch))
    
    fig.savefig('./outputs/{0}/figures/Graphs/Graphs{1}'.format(model_name, epoch))
    plt.close(fig)
    
    return figs_data

def load_checkpoints(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(flags.checkpoint_dir):
            os.mkdir(flags.checkpoint_dir)
    return saver

def main(argv):
    manager = DataManager(input_name)
    manager.load()
    
    sess = tf.Session()
    
    model = VAE(gamma=flags.gamma,
                capacity_limit=flags.capacity_limit,
                capacity_change_duration=flags.capacity_change_duration,
                learning_rate=flags.learning_rate,
                channels=channels)
    
    sess.run(tf.global_variables_initializer())
    
    saver = load_checkpoints(sess)
    
    if flags.training:
        # Train
        train(sess, model, manager, saver)
    else:
        reconstruct_check_images = manager.get_random_images(10)
        # Image reconstruction check
        reconstruct_check(sess, model, reconstruct_check_images)
        # Disentangle check
        disentangle_check(sess, model, manager)

if __name__ == '__main__':
    tf.app.run()
