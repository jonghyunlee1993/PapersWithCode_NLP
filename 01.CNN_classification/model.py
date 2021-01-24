import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1d(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx, mode=0):
        super().__init__()
        self.mode = mode

        self.static_embedding_layer    = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx, freeze=True)
        self.nonstatic_embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx, freeze=False)

        self.convolutional_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.fully_connected_layers = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.droupout_layers = nn.Dropout(dropout)

        if self.mode == 0:
            print("Static mode")
        elif self.mode == 1:
            print("Nonstatic mode")
        else:
            print("Static + Nonstatic mode")

    def static_mode_network(self, text):
        embedded = self.static_embedding_layer(text)
        embedded = embedded.permute(0, 2, 1)
        conved   = [F.tanh(conv(embedded)) for conv in self.convolutional_layers]
        pooled   = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat      = self.dropout_layers(torch.cat(pooled, dim=1))

        return cat

    def nonstatic_mode_network(self, text):
        embedded = self.nonstatic_embedding_layer(text)
        embedded = embedded.permute(0, 2, 1)
        conved = [F.tanh(conv(embedded)) for conv in self.convolutional_layers]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout_layers(torch.cat(pooled, dim=1))

        return cat

    def static_and_nonstaticz_mode_network(self, text):
        embedded_1 = self.static_embedding_layer(text)
        embedded_1 = embedded_1.permute(0, 2, 1)
        conved_1   = [F.tanh(conv(embedded_1)) for conv in self.convolutional_layers]
        pooled_1   = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_1]

        embedded_2 = self.nonstatic_embedding_layer(text)
        embedded_2 = embedded_2.permute(0, 2, 1)
        conved_2 = [F.tanh(conv(embedded_2)) for conv in self.convolutional_layers]
        pooled_2 = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_2]

        cat = self.dropout_layers(torch.cat([pooled_1, pooled_2], dim=1))

        return cat

    def forward(self, text):
        if self.mode == 0:
            output = self.static_mode_network(text)
            return self.fully_connected_layers(output)
        elif self.mode == 1:
            output = self.nonstatic_mode_network(text)
            return self.fully_connected_layers(output)
        else:
            output = self.static_and_nonstaticz_mode_network(text)
            return self.fully_connected_layers(output)