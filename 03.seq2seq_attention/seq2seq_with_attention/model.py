import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, use_bidirection=True):
        super().__init__()
        self.use_bidirection = use_bidirection
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        if self.use_bidirection:
            self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        else:
            self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        
        if self.use_bidirection:        
            hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        else:
            hidden = torch.tanh(self.fc(hidden[-2, :, :]))

        return outputs, hidden
    

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)
    
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, maxout_size=None):
        super().__init__()

        self.maxout_size = maxout_size
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        if self.maxout_size:
            self.maxout = Maxout((enc_hid_dim * 2) + dec_hid_dim + emb_dim, dec_hid_dim, maxout_size)
            self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        else:
            self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)      
        embedded = self.dropout(self.embedding(input))
        
        attention = self.attention(hidden, encoder_outputs)
        attention = attention.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(attention, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        if self.maxout_size:
            maxout_input = torch.cat((output, weighted, embedded), dim = 1)
            output = self.maxout(maxout_input).squeeze(0)
            prediction = self.fc_out(outut)
        else:
            prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        return prediction, hidden.squeeze(0)
    
    
class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)
 
    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)

        return m
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):      
        src_length, batch_size = src.shape
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0,:]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            
            input = trg[t] if random.random() < teacher_forcing_ratio else output.argmax(1)

        return outputs
    
    
def init_weights(model):
    for name, param in model.named_parameters():
        if 'rnn.weight' in name:
            nn.init.orthogonal_(param.data)
        elif 'attention.v' in name:
            nn.init.zeros_(param.data)
        elif 'attention' in name:
            nn.init.normal_(param.data, 0, 0.001 ** 2)
        elif 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01 ** 2)
        else:
            nn.init.constant_(param.data, 0)
            

def count_parameters(model):
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'The model has {param:,} trainable parameters')

