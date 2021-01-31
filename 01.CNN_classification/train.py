import torch
import random
import torchtext
import numpy as np

from src.data_loader import *
from src.preprocessor import Preprocessor
from src.model import PolarCNN
from src.util import *


import argparse

print(f"PyTorch version: {torch.__version__}\nTorchtext version: {torchtext.__version__}")

class Main:
    def __init__(self, args):
        self.args = args
        self.args.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = Preprocessor(self.args)

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        # torch.backend.cudnn.deterministic = True


    def get_dataset_info(self):
        self.vocab_size = self.dataset.vocab.idx
        self.pad_idx    = self.dataset.vocab(self.dataset.vocab.PAD_TOKEN)


    def run(self):
        self.set_seed()

        reviews, word2index = self.preprocessor.build_data()
        self.dataset = load_tabular_dataset(self.args.procssed_csv)



        if self.args.cv_mode:
            for cv_index in range(self.args.cv_split):
                self.train_iterator, self.valid_iterator = data_load_with_cv(self.dataset)

        else:
            self.train_iterator, self.valid_iterator = data_load_without_cv(self.dataset)
            self.get_dataset_info()
            self.pretrained_embedding = load_pretrained_word2vec(self.args.data_path)
            self.model = PolarCNN(self.vocab_size, self.pad_idx, self.args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # preprocessor
    parser.add_argument('--data_path', type=str, default='data', help='data folder')
    parser.add_argument('--pos_data', type=str, default='rt-polarity.pos', help='file name of positive raw data')
    parser.add_argument('--neg_data', type=str, default='rt-polarity.neg', help='file name of negative raw data')
    parser.add_argument('--preprocessed_data', type=str, default='polarity_data.pickle', help='file name of processed data')

    # model
    parser.add_argument('--cv_mode', type=bool, default=False, help="cross validation: [False] / True")
    parser.add_argument('--cv_split', type=int, default=10, help='number of split for cross validation')
    parser.add_argument('--device', type=str, default='cpu', help='[cpu] / cuda')
    parser.add_argument('--cnn_mode', type=str, default='static', help="[static] / nonstatic / multi")
    parser.add_argument('--max_norm_scaling', type=bool, default=False, help='[False] / True')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for model training')
    parser.add_argument('--epochs', type=int, default=10, help='model training epochs')
    parser.add_argument('--embedding_dim', type=int, default=300, help='pretrained embedding vector dimension')
    parser.add_argument('--filter_number', type=int, default=100, help='number of output of each filters')
    parser.add_argument('--filter_size', type=list, default=[3, 4, 5], help='size of each filters')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='rate of dropout')
    parser.add_argument('--seed', type=int, default=1234, help='seed number, default: 1234')
    parser.add_argument('--pretrained_model', type=str, default='word2vec', help='[word2vec] / glove')

    args = parser.parse_args()

    main = Main(args)
    main.run()