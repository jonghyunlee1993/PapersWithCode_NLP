import torch
import random
import torchtext
import numpy as np

from src.data_loader import *
from src.preprocessor import Preprocessor
from src.model import PolarCNN
from src.util import *
from src.torch_util import *

import argparse

print(f"PyTorch version: {torch.__version__}\nTorchtext version: {torchtext.__version__}")

class Predictor:
    def __init__(self, args):
        self.args = args
        self.args.device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_param_fname  = os.path.join("weights", f"best_model_{self.args.cnn_mode}.pt")
        self.preprocessor       = Preprocessor(self.args)

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        # torch.backend.cudnn.deterministic = True

    def get_vocab_info(self):
        self.vocab_size = len(self.TEXT.vocab)
        self.pad_index = self.TEXT.vocab.stoi['<pad>']
        self.unk_index = self.TEXT.vocab.stoi['<unk>']

        print(f"input dim: {self.vocab_size}\npad index: {self.pad_index}\nunk index: {self.unk_index}")

    def run(self):
        self.set_seed()

        reviews, word2index = self.preprocessor.build_data()


        processed_csv_fname = os.path.join(self.args.data_path, self.args.processed_csv)
        self.TEXT, self.train_iterator, self.valid_iterator = data_load_without_cv(processed_csv_fname, self.args)
        self.get_vocab_info()
        word2vec_index, word2vec_vector = load_pretrained_word2vec(self.args.data_path, self.TEXT)

        self.TEXT.vocab.set_vectors(word2vec_index, torch.from_numpy(word2vec_vector).float().to(self.args.device),
                                    self.args.embedding_dim)

        self.model = PolarCNN(self.vocab_size, self.pad_index, self.args).to(self.args.device)
        self.critierion = torch.nn.BCEWithLogitsLoss().to(self.args.device)

        # for prediction
        if os.path.isfile(self.model_param_fname):
            self.model.load_state_dict(torch.load(self.model_param_fname))
            probability, predicted_label = predict(self.TEXT, self.args, self.model)
            print_predict_log(self.args, probability, predicted_label)
        else:
            print(f"model parameter file {self.model_param_fname} was not exist")
            print(f"for sentence prediction, please train your model first")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # preprocessor
    parser.add_argument('--data_path', type=str, default='data', help='data folder')
    parser.add_argument('--pos_data', type=str, default='rt-polarity.pos', help='file name of positive raw data')
    parser.add_argument('--neg_data', type=str, default='rt-polarity.neg', help='file name of negative raw data')
    parser.add_argument('--processed_pickle', type=str, default='polarity_data.pickle', help='file name of processed data')
    parser.add_argument('--processed_csv', type=str, default='polarity_data.csv', help='file name of processed data')

    # model
    parser.add_argument('--device', type=str, default='cpu', help='[cpu] / cuda')
    parser.add_argument('--cnn_mode', type=str, default='multi', help="[static] / nonstatic / multi")
    parser.add_argument('--embedding_dim', type=int, default=300, help='pretrained embedding vector dimension')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for model prediction')
    parser.add_argument('--filter_number', type=int, default=100, help='number of output of each filters')
    parser.add_argument('--filter_size', type=list, default=[3, 4, 5], help='size of each filters')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='rate of dropout')
    parser.add_argument('--output_dim', type=int, default=1, help='dimension of output')
    parser.add_argument('--seed', type=int, default=1234, help='seed number, default: 1234')
    parser.add_argument('--pretrained_model', type=str, default='word2vec', help='[word2vec] / glove')

    parser.add_argument('--input_sent', type=str, default='This film is totally terrible!', help='input sentence to predict')

    args = parser.parse_args()

    trainer = Predictor(args)
    trainer.run()