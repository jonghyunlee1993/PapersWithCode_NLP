import os
import re
import time
import json
import torch
import random
import codecs
import gensim
import torchtext
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

import argparse

print(f"PyTorch version: {torch.__version__}\nTorchtext version: {torchtext.__version__}")


def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backend.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0,
                        help='choose static / nonstatic mode {0: static, 1: nonstatic, 2: static + nonstatic mode}')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for model training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='model training epochs')
    parser.add_argument('--embedding-dim', type=int, default=300,
                        help='pretrained embedding vector dimension')
    parser.add_argument('--filter-number', type=int, default=100,
                        help='number of output of each filters')
    parser.add_argument('--filter-size', type=list, default=[3, 4, 5],
                        help='size of each filters')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='rate of dropout')
    parser.add_argument('--seed', type=int, default=1234,
                        help='seed number, default: 1234')
    parser.add_argument('--vector', type=str, default='w2v',
                        help='pretrained word embedding model, w2v or glove')


