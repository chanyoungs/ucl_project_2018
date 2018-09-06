# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import norm
import sys

from model import MODEL
from data_manager import DataManager

model_summary = {}
model_summary_txt = ''

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("data", "test", "data")
flags.DEFINE_string("outputs", "outputs1", "outputs folder index")
flags.DEFINE_string("model_name", "test", "model name") # default 20000
flags.DEFINE_integer("epoch_size", 30, "epoch size") # default 20000
flags.DEFINE_integer("latent_size", 20, "latent size")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_float("gamma", 1, "gamma param for latent loss") # default 100
flags.DEFINE_float("capacity_limit", 0.0, "encoding capacity limit param for latent loss") # default 20
flags.DEFINE_integer("capacity_change_duration", 100000, "encoding capacity change duration") # default 100000
flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
flags.DEFINE_boolean("training", True, "training or not")
flags.DEFINE_boolean("test", False, "training or not")

checkpoint_dir = f"./{FLAGS.outputs}/{FLAGS.model_name}/checkpoints"
log_file = f"./{FLAGS.outputs}/{FLAGS.model_name}/logs"

if FLAGS.data == "test":
    if not FLAGS.test:
        user_input = input("No data chosen. Run test mode?([y]=Yes/n=Abort)")
        while True:
            if user_input == "y" or user_input == "":
                break
            elif user_input == "n":
                print("Aborting...")
                quit()
            else:
                user_input = input("Please type 'y' or 'n'. Running test mode?[y=Yes/n=Abort]")
    print("Running test mode...")

# Make directories
path = os.path.join(FLAGS.outputs, FLAGS.model_name)
directories = [
    ['logs'],
    ['figs_data'],
    ['checkpoints'],
    ['figures', 'Graphs'],
    ['figures', 'Disentanglements'],
    ['figures', 'Reconstructions'],
    ['figures', 'Latest']
]

for paths in directories:
    full_dir = path
    for folder in paths:
        full_dir = os.path.join(full_dir, folder)

    if not os.path.isdir(full_dir):
        print(f'{full_dir} does not exist. Creating path...')
        os.makedirs(full_dir)
    else:
        print(f'{full_dir} already exists')

# Save and print model summary
flags_dic = FLAGS.flag_values_dict()
for key in flags_dic.keys():
    if not key =='training':
        model_summary_txt += f'{key}: {flags_dic[key]} \n'

with open(os.path.join(path, 'figs_data', 'model_summary.txt'), 'w') as text_writer:
    text_writer.write(model_summary_txt)

print("\n\n###########Model Summary##########\n")
print(model_summary_txt)
print("###########Model Summary##########\n\n")

# Save model summary as dictionary
model_summary['gamma'] = FLAGS.gamma
model_summary['batch_size'] = FLAGS.batch_size
model_summary['capacity_limit'] = FLAGS.capacity_limit
model_summary['capacity_change_duration'] = FLAGS.capacity_change_duration
model_summary['learning_rate'] = FLAGS.learning_rate

np.save(os.path.join(path, 'figs_data', 'model_summary.npy'), model_summary)

def train(sess, model, manager, saver):

    summary_writer = tf.summary.FileWriter(log_file, sess.graph)
    
    n_samples = manager.sample_size
    
    reconstruct_check_images = manager.get_random_images(10)
    
    indices = list(range(n_samples))
    
    if not os.path.isfile(os.path.join(path, 'figs_data', 'figs_data.npy')):
        figs_data = {
            'epoch' : 1,
            'step' : 0,
            'latent_vars' : [],
            'latent_means' : [],
            'losses_r' : [], # Reconstruction losses over epochs for plot
            'losses_l' : [], # Latent losses over epochs for plot
            'losses_l_w' : [], # Weighted latent losses over epochs for plot
            'losses_t' : [], # Final losses over epochs for plot
            'disentangled_metric' : []
        }
    else:
        figs_data = np.load(os.path.join(path, 'figs_data', 'figs_data.npy'))
        figs_data = figs_data.item()
    
    # Training cycle
    while figs_data['epoch'] <= FLAGS.epoch_size:
        # Shuffle image indices
        random.shuffle(indices)
        
        total_batch = n_samples // FLAGS.batch_size
        
        # Loop over all batches
        for i in range(total_batch):
            # Generate image batch
            batch_indices = indices[FLAGS.batch_size*i : FLAGS.batch_size*(i+1)]
            batch_xs = manager.get_images(batch_indices)
            
            # Fit training using batch data
            reconstr_loss, latent_loss, latent_loss_weighted, total_loss, summary_str = model.partial_fit(sess, batch_xs, figs_data['step'])
            
            figs_data['step'] += 1
            
        summary_writer.add_summary(summary_str, figs_data['epoch'])
        
        figs_data['losses_r'].append(reconstr_loss)
        figs_data['losses_l'].append(latent_loss)
        figs_data['losses_l_w'].append(latent_loss_weighted)
        figs_data['losses_t'].append(total_loss)
        
        # Image reconstruction check & disentanglement check
        figs_data, dis_met = plot_figures(sess, model, reconstruct_check_images, manager, figs_data)
        
        print(f"Epoch: {figs_data['epoch']} Loss_R: {reconstr_loss} Loss_L: {latent_loss}, Dis_Met: {dis_met}")

        # save to csv
        save_to_csv(figs_data)
        # Save checkpoint
        saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step = figs_data['epoch'])

        # save to numpy
        np.save(os.path.join(path, 'figs_data', 'figs_data.npy'), figs_data)

        figs_data['epoch'] += 1

def save_to_csv(figs_data):
    dic = {'_epoch': range(1, figs_data['epoch'] + 1)}
    for key in figs_data.keys():
        if not key in ('epoch', 'step', 'latent_vars', 'latent_means'):
            dic[key] = figs_data[key]

    with open(os.path.join(path, 'figs_data', 'model_summary.csv'), 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dic.keys())
        writer.writerows(zip(*dic.values()))

        
def plot_figures(sess, model, images, manager, figs_data):
    epoch = figs_data['epoch']

    # Conitional imshow function
    if manager.n_channels == 3:
        imshow2 = lambda image: plt.imshow(image)
    else:
        imshow2 = lambda image: plt.imshow(image, cmap='Greys_r')

    #################### Check image reconstruction
    x_reconstruct = model.reconstruct(sess, images)
    
    fig = plt.figure(facecolor='grey', figsize=(10, 4))
    for i in range(len(images)):
        image_shape = (64, 64) if images[0].shape[0] == 64*64 else (64, 64, 3)
        
        org_img = images[i].reshape(image_shape)
        org_img = org_img.astype(np.float32)
        
        reconstr_img = x_reconstruct[i].reshape(image_shape)
        
        plt.subplot(2, 10, i+1)
        imshow2(org_img)
        plt.axis('off')
        
        plt.subplot(2, 10, i+11)
        imshow2(reconstr_img)
        plt.axis('off')
        
    fig.suptitle(f'Original images(above), Reconstruction images(below) Epoch: {epoch}')
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Reconstructions/reconstruction{epoch}')
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Reconstructions/reconstruction')
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Latest/reconstruction')
    plt.close(fig) 
    
    ###################### Latent traversal
    img = manager.imgs[0].reshape(np.prod(image_shape))
    
    batch_xs = [img]
    z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
    z_sigma_sq = np.exp(z_log_sigma_sq)[0]
    
    figs_data['latent_vars'].append(z_sigma_sq.reshape(FLAGS.latent_size, 1))
    figs_data['latent_means'].append(z_mean.reshape(FLAGS.latent_size, 1))
    
    latent_vars_graph = np.concatenate(figs_data['latent_vars'], axis=1)
    latent_means_graph = np.concatenate(figs_data['latent_means'], axis=1)
    
    # Save disentangled images
    z_m = z_mean[0]
    
    ind = np.argsort(np.array(figs_data['latent_vars'][-1]).flatten())
    ind_inv = np.argsort(ind)
  
    fig = plt.figure(figsize=(10,FLAGS.latent_size), facecolor='grey')
    for target_z_index in range(FLAGS.latent_size):
        for ri in range(10):
            value = -3.0 + (6.0 / 9.0) * ri
            z_mean2 = np.zeros((1, FLAGS.latent_size))
            for i in range(FLAGS.latent_size):
                if( i == target_z_index ):
                    z_mean2[0][i] = value
                else:
                    z_mean2[0][i] = z_m[i]
            reconstr_img = model.generate(sess, z_mean2)
            rimg = reconstr_img[0].reshape(image_shape)
            
            plt.subplot(FLAGS.latent_size, 10, ind_inv[target_z_index]*10 + ri+1)
            imshow2(rimg)
            plt.axis('off')
      
    fig.suptitle(f'Latent space traversal Epoch: {epoch}')
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Disentanglements/latent_space{epoch}')
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Disentanglements/latent_space')
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Latest/latent_space')
    plt.close(fig)
  
    ############### Graphs
    fig = plt.figure(figsize=(20, 20))

    # Latent variances
    plt.subplot(321)
    for v in range(FLAGS.latent_size):
        plt.plot(range(latent_vars_graph.shape[1]), latent_vars_graph[ind[v],:], label=v+1)
    plt.legend()
    plt.title(f'Latent variances Epoch: {epoch}')

    # Latent means
    plt.subplot(322)
    for v in range(FLAGS.latent_size):
        plt.plot(range(latent_means_graph.shape[1]), latent_means_graph[ind[v],:], label=v+1)
    plt.legend()
    plt.title(f'Latent means Epoch: {epoch}')

    # Latent Gaussians
    plt.subplot(323)
    x = np.linspace(np.min(latent_means_graph[:, -1]) - 3 * np.max(latent_vars_graph[:, -1]), np.max(latent_means_graph[:, -1]) + 3 * np.max(latent_vars_graph[:, -1]), 300)
    for v in range(FLAGS.latent_size):
        plt.plot(x, norm.pdf(x, latent_means_graph[ind[v],-1], np.sqrt(latent_vars_graph[ind[v],-1])), label=v+1)
    plt.legend()
    plt.title(f'Latent Gaussians at Epoch: {epoch}')

    # Losses
    plt.subplot(324)
    plt.plot(range(len(figs_data['losses_r'])), figs_data['losses_r'], label='Reconstruction')
    plt.plot(range(len(figs_data['losses_l'])), figs_data['losses_l'], label='Latent')
    plt.plot(range(len(figs_data['losses_l_w'])), figs_data['losses_l_w'], label='Weighted latent')
    plt.plot(range(len(figs_data['losses_t'])), figs_data['losses_t'], label='Total')
    plt.legend()
    plt.title(f'Losses Epoch: {epoch}')
        
    # Disentanglement metric
    images_total = manager.imgs[:10000]
    images_total = images_total.reshape(images_total.shape[0], -1)
    batch_xs = images_total
    z_mean_total, z_log_sigma_sq = model.transform(sess, batch_xs)
    z_mean_total_var = np.var(z_mean_total, axis=0)
    
    images_dis = manager.imgs_dis
    latents_gt = manager.latents_gt
    no_latents = int(np.max(latents_gt) + 1) # Number of original latents in the
    
    votes = np.zeros((FLAGS.latent_size, no_latents))
    # print('Counting votes...')
    for n in range(images_dis.shape[0]):
        batch_xs = images_dis[n]
        z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
        normalised = np.divide(z_mean, z_mean_total_var)
        normalised_var = np.var(normalised, axis=0)
        argmin = np.argmin(normalised_var)
        votes[argmin, int(latents_gt[n])] += 1
        
    # print('Calculating accuracy...')
    dis_met = 0
    for n in range(FLAGS.latent_size):
        dis_met += np.max(votes[n])
        
    dis_met *= 100 / images_dis.shape[0]
    # print('Accuracy: {accuracy}%'
    
    figs_data['disentangled_metric'].append(dis_met)
    
    plt.subplot(325)
    plt.plot(range(len(figs_data['disentangled_metric'])), figs_data['disentangled_metric'])
    plt.title(f'Disentanglement Metric Epoch: {epoch}')
    
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Graphs/Graphs{epoch}')
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Graphs/Graphs')
    fig.savefig(f'./{FLAGS.outputs}/{FLAGS.model_name}/figures/Latest/Graphs')
    plt.close(fig)
    
    return figs_data, dis_met

def load_checkpoints(sess):
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print(f"loaded checkpoint: {checkpoint.model_checkpoint_path}")
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
    return saver

def main(argv):
    manager = DataManager(FLAGS.data)
    manager.load()

    sess = tf.Session()

    normaliser = (FLAGS.latent_size / 10) * ((64 * 64) / manager.input_size)
    
    model = MODEL(latent_size=FLAGS.latent_size,
                  gamma=normaliser*FLAGS.gamma,
                  capacity_limit=FLAGS.capacity_limit,
                  capacity_change_duration=FLAGS.capacity_change_duration,
                  learning_rate=FLAGS.learning_rate,
                  n_channels=manager.n_channels
    )
    
    sess.run(tf.global_variables_initializer())
    
    saver = load_checkpoints(sess)
    
    if FLAGS.training:
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
