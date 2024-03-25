import torch
from dataset import DataSet
import argparse
import os

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, default=None)
parser.add_argument('--dimensions', nargs="*", type=int, default=[3, 3, 3],
                    help='Number of features for every perceptual dimension')
parser.add_argument('--game_size', type=int, default=10,
                    help='Number of target/distractor objects')
parser.add_argument("--save", type=bool, default=True)

args = parser.parse_args()

# prepare folder for saving
if args.path:
    if not os.path.exists(args.path + 'data/'):
        os.makedirs(args.path + 'data/')
else:
    if not os.path.exists('data/'):
        os.makedirs('data/')

data_set = DataSet(args.dimensions,
                   game_size=args.game_size,
                   device='cpu')

if args.path:
    path = (args.path + 'data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ').ds')
else:
    path = ('data/dim(' + str(len(args.dimensions)) + ',' + str(args.dimensions[0]) + ').ds')

if args.save:
    with open(path, "wb") as f:
        torch.save(data_set, f)
        print("Data set is saved as: " + path)
