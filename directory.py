import os
import sys

path = os.path.join('outputs', sys.argv[1])

if not os.path.isdir(path):
    print('Path does not exist. Creating path...')
    os.makedirs(
        os.path.join(path, 'logs'))
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
