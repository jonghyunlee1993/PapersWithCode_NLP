import torch
import random
import torchtext
import numpy as np

from src.preprocessor import Preprocessor

import argparse

print(f"PyTorch version: {torch.__version__}\nTorchtext version: {torchtext.__version__}")

class Main:
    def __init__(self, args):
        self.args = args
        self.preprocessor = Preprocessor(self.args)


    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        # torch.backend.cudnn.deterministic = True

    def run(self):
        self.set_seed()

        Preprocessor()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # preprocessor
    parser.add_argument('--data_path', type=str, default='data', help='data folder')
    parser.add_argument('--pos_data', type=str, default='rt-polarity.pos', help='file name of positive raw data')
    parser.add_argument('--neg_data', type=str, default='rt-polarity.neg', help='file name of negative raw data')
    parser.add_argument('--preprocessed_data', type=str, default='polarity_data.pickle', help='file name of processed data')

    # model
    parser.add_argument('--mode', type=str, default='static', help="[static] / nonstatic / multi")
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for model training')
    parser.add_argument('--epochs', type=int, default=10, help='model training epochs')
    parser.add_argument('--embedding-dim', type=int, default=300, help='pretrained embedding vector dimension')
    parser.add_argument('--filter-number', type=int, default=100, help='number of output of each filters')
    parser.add_argument('--filter-size', type=list, default=[3, 4, 5], help='size of each filters')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='rate of dropout')
    parser.add_argument('--seed', type=int, default=1234, help='seed number, default: 1234')
    parser.add_argument('--vector', type=str, default='word2vec', help='[word2vec] / glove')

    args = parser.parse_args()

    main = Main(args)
    main.run()