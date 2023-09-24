import argparse
from src.model import RetBlock

args = argparse.ArgumentParser()
args.add_argument('--lr', type=float, default=0.01)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--epochs', type=int, default=10)
args.add_argument('--dataset', type=str, default='the-stack')

# load data

# create model

# train model
