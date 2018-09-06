import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('key', help='Write the key of the value you wish to retrieve')
parser.add_argument('-o', '--outputs', help='Which output do you want? Int: 1, 2 or 3', type=int)
parser.add_argument('-p', '--path', help='./path/figs_data/figs_data.npy')
parser.add_argument('-e', '--epoch', help='Write the epoch of the value you wish to retrieve', type=int)

args = parser.parse_args()

path = f'./outputs{args.outputs}/{args.path}/figs_data/figs_data.npy'

dic_np = np.load(path)
dic = dic_np.item()
if args.epoch:
    print(f'{args.key} at epoch {args.epoch}: {dic[args.key][args.epoch]}')
else:
    print(f'{args.key}: {dic[args.key]}')

