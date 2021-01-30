import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarCNN(nn.Module):
    def __init__(self, vocab_size, pad_idx, args):
        super().__init__()
        self.args = args
        print(f"CNN mode: {self.args.mode}")

        self.base_embedding       = nn.Embedding(vocab_size, args.embedding_dim, padding_idx=pad_idx)
        self.additional_embedding = nn.Embedding(vocab_size, args.embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=args.n_filters,
                      kernel_size=(fs, args.embedding_dim))
            for fs in args.filter_sizes
        ])

        self.fc = nn.Linear(len(args.filter_sizes) * args.n_filters, args.output_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, text):
        if self.args.mode == 'static' or self.args.mode == 'nonstatic':
            base_embedded = self.base_embedding(text)
            base_embedded = base_embedded.unsqueeze(1)
            base_conved = [F.relu(conv(base_embedded)).squeeze(3) for conv in self.convs]
        elif self.args.mode == 'multi':
            base_embedded = self.base_embedding(text)
            base_embedded = base_embedded.unsqueeze(1)
            base_conved = [F.relu(conv(base_embedded)).squeeze(3) for conv in self.convs]

            additional_embedded = self.additional_embedding(text)
            additional_embedded = additional_embedded.unsqueeze(1)
            additional_conved = [F.relu(conv(additional_embedded)).squeeze(3) for conv in self.convs]

            base_conved = [base_conved[i] + additional_conved[i] for i in range(len(base_conved))]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in base_conved]
        cat = self.dropout(torch.cat(pooled, dim=1))

        return self.fc(cat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Model Builder')
    parser.add_argument('--mode', type=str, default='static', help="[static] / nonstatic / multi")
    parser.add_argument('--embedding_dim', type=int, default=300, help='dimesion of embedding')
    parser.add_argument('--n_filters', type=int, default=100, help='number of filter for each convolutional layers')
    parser.add_argument('--filter_sizes', type=list, default=[3, 4, 5], help='size of each filter. similar to n-gram')
    parser.add_argument('--output_dim', type=int, default=1, help='dimision of output. For predict polarity, put 1')
    parser.add_argument('--dropout', type=float, default=0.5, help='rate of dropout')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PolarCNN(vocab_size=1000, pad_idx=0, args=args).to(device)
    sample = torch.randint(20, (3, 5)).to(device)
    res = model(sample)

    print(res.shape)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')