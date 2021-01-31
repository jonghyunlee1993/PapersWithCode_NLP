import torch
import random
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from torchtext.data import Field, LabelField, TabularDataset


def load_tabular_dataset(fname, fix_length=56):
    TEXT   = Field(sequential=True, tokenize=str.split, batch_first=True, fix_length=fix_length, lower=True)
    LABEL  = LabelField(sequential=False, dtype=torch.float)
    FIELDS = [('label', LABEL), ('text', TEXT)]

    dataset = TabularDataset(fname, fields=FIELDS, format='csv', skip_header=True)

    return dataset

def data_load_with_cv(dataset, args, cv_index=0, seed=1234, n_split=10):
    kf = KFold(n_splits=n_split, random_state=seed, shuffle=True)

    for i, (train_index, test_index) in enumerate(kf.split(np.arange(len(dataset)))):
        # print(f"train_idnex: {len(train_index)} and test_index: {len(test_index)}")

        if i == cv_index:
            print("-" * 30)
            print(f"subset index: {i:02}")
            print("")
            train_subset   = Subset(dataset, train_index)
            train_iterator = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_padding)

            test_subset    = Subset(dataset, test_index)
            test_iterator  = DataLoader(test_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_padding)

            break

    return train_iterator, test_iterator


def data_load_without_cv(dataset, seed=1234, split_ratio=0.9):
    train_dataset, valid_dataset = dataset.split(random_state=random.seed(seed), split_ratio=split_ratio)

    return train_dataset, valid_dataset


def collate_fn_padding(batch):
    batch_x = [t[0] for t in batch]
    batch_y = [t[1] for t in batch]
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, padding_value=1,  batch_first=True).to(torch.int64)
    batch_y = torch.FloatTensor(batch_y)

    return batch_x, batch_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Loader')
    parser.add_argument('--processed_csv', type=str, default='../data/polarity_data.csv', help="")
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    dataset = load_tabular_dataset(args.processed_csv)
    # train_dataset, test_dataset = data_load_without_cv(dataset)
    # print(f"train_dataset: {len(train_dataset)}\ntest_dataset: {len(test_dataset)}")

    for i in range(10):
        train_dataset, test_dataset = data_load_with_cv(dataset, args, cv_index=i)
        print(f"train_dataset: {len(train_dataset)}\ntest_dataset: {len(test_dataset)}")