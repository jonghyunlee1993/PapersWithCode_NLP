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

class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_param_fname  = os.path.join("weights", "best_model.pt")
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

        if self.args.cv_mode:
            for cv_index in range(self.args.cv_split):
                self.train_iterator, self.valid_iterator = data_load_with_cv(self.dataset, self.args)

        else:
            processed_csv_fname = os.path.join(self.args.data_path, self.args.processed_csv)
            self.TEXT, self.train_iterator, self.valid_iterator = data_load_without_cv(processed_csv_fname, self.args)
            self.get_vocab_info()
            word2vec_index, word2vec_vector = load_pretrained_word2vec(self.args.data_path, self.TEXT)

            self.TEXT.vocab.set_vectors(word2vec_index, torch.from_numpy(word2vec_vector).float().to(self.args.device),
                                        self.args.embedding_dim)

            self.model = PolarCNN(self.vocab_size, self.pad_index, self.args).to(self.args.device)

            if self.args.cnn_mode == 'static':
                self.model.base_embedding.weight.data.copy_(self.TEXT.vocab.vectors)
                proc_special_token(self.model.base_embedding, self.TEXT, self.args.embedding_dim)
                self.model.base_embedding.weight.requires_grad = False
            elif self.args.cnn_mode == 'nonstatic':
                self.model.base_embedding.weight.data.copy_(self.TEXT.vocab.vectors)
                proc_special_token(self.model.base_embedding, self.TEXT, self.args.embedding_dim)
                self.model.base_embedding.weight.requires_grad = True
            elif self.args.cnn_mode == 'multi':
                self.model.base_embedding.weight.data.copy_(self.TEXT.vocab.vectors)
                proc_special_token(self.model.base_embedding, self.TEXT, self.args.embedding_dim)
                self.model.base_embedding.weight.requires_grad = False
                self.model.additional_embedding.weight.data.copy_(self.TEXT.vocab.vectors)
                proc_special_token(self.model.additional_embedding, self.TEXT, self.args.embedding_dim)
                self.model.additional_embedding.weight.requires_grad = True
            else:
                print("choose between static / nonstatic / multi")
                raise()

            # define optimizer & criterion
            self.optimizer  = torch.optim.Adadelta(self.model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
            self.critierion = torch.nn.BCEWithLogitsLoss().to(self.args.device)

            best_valid_loss = float('inf')
            for epoch in range(self.args.epochs):
                start_time = get_time()
                train_loss, train_acc = train(self.model, self.train_iterator, self.optimizer, self.critierion)
                valid_loss, valid_acc = evaluate(self.model, self.valid_iterator, self.critierion)
                end_time = get_time()

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    save_model_param(self.model_param_fname, self.model)

                print_training_log(epoch, start_time, end_time, train_loss, train_acc, valid_loss, valid_acc)

        # for evaluattion
        self.model = torch.load_state_dict(torch.load(self.model_param_fname))
        start_time = get_time()
        test_loss, test_acc = evaluate(self.model, self.valid_iterator, self.critierion)
        end_time = get_time()
        print_evaluation_log(start_time, end_time, test_loss, test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # preprocessor
    parser.add_argument('--data_path', type=str, default='data', help='data folder')
    parser.add_argument('--pos_data', type=str, default='rt-polarity.pos', help='file name of positive raw data')
    parser.add_argument('--neg_data', type=str, default='rt-polarity.neg', help='file name of negative raw data')
    parser.add_argument('--processed_pickle', type=str, default='polarity_data.pickle', help='file name of processed data')
    parser.add_argument('--processed_csv', type=str, default='polarity_data.csv', help='file name of processed data')

    # model
    parser.add_argument('--cv_mode', type=bool, default=False, help="cross validation: [False] / True")
    parser.add_argument('--cv_split', type=int, default=10, help='number of split for cross validation')
    parser.add_argument('--device', type=str, default='cpu', help='[cpu] / cuda')
    parser.add_argument('--cnn_mode', type=str, default='multi', help="[static] / nonstatic / multi")
    parser.add_argument('--max_norm_scaling', type=bool, default=False, help='[False] / True')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for model training')
    parser.add_argument('--epochs', type=int, default=10, help='model training epochs')
    parser.add_argument('--embedding_dim', type=int, default=300, help='pretrained embedding vector dimension')
    parser.add_argument('--filter_number', type=int, default=100, help='number of output of each filters')
    parser.add_argument('--filter_size', type=list, default=[3, 4, 5], help='size of each filters')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='rate of dropout')
    parser.add_argument('--output_dim', type=int, default=1, help='dimension of output')
    parser.add_argument('--seed', type=int, default=1234, help='seed number, default: 1234')
    parser.add_argument('--pretrained_model', type=str, default='word2vec', help='[word2vec] / glove')

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()