import time
import torch
import random

import config
import numpy as np

from seq2seq_with_attention.utils import epoch_time, print_loss
from seq2seq_with_attention.seq2seq_model import init_weights, count_parameters
from seq2seq_with_attention.seq2seq_model import Encoder, Decoder, Seq2Seq, Attention
from seq2seq_with_attention.custom_dataloader import CustomDataloader

class Trainer:
    def __init__(self) -> None:
        self.set_seed()
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.BATCH_SIZE = config.BATCH_SIZE
        
        self.dataloader = CustomDataloader(self.BATCH_SIZE, self.DEVICE)

        
    def run(self):
        self.dataloader.run()
        self.INPUT_DIM  = len(self.dataloader.SRC.vocab)
        self.OUTPUT_DIM = len(self.dataloader.TRG.vocab)
        
        self.define_model()
        
        best_valid_loss = float('inf')
        for epoch in range(config.TRAIN_EPOCHS):
            start_time = time.time()
            train_loss = self.train()
            valid_loss = self.evaluate()
            end_time   = time.time()
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'tut3-model.pt')
            
            print(f"Epoch Num: {epoch}")
            epoch_time(start_time, end_time)
            print_loss(train_loss, valid_loss)
            
            
        
    def set_seed(self):
        seed = config.SEED
        random.seed(config.seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    
    def define_model(self):
        self.encoder   = Encoder(self.INPUT_DIM, config.ENC_EMB_DIM, config.ENC_HID_DIM, config.DEC_HID_DIM, config.ENC_DROPOUT, config.USE_BIDIRECTION)
        self.attention = Attention(config.ENC_HID_DIM, config.DEC_HID_DIM)
        self.decoder   = Decoder(self.OUTPUT_DIM, config.DEC_EMB_DIM, config.ENC_HID_DIM, config.DEC_HID_DIM, config.DEC_DROPOUT, self.attention, config.MAXOUT_SIZE)
        self.model     = Seq2Seq(self.encoder, self.decoer, self.DEVICE).to(self.DEVICE)

        self.model.apply(init_weigths)
        count_parameters(self.model)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.dataloader.TRG.vocab.stoi[self.dataloader.TRG.pad_token])
    
        
    def train(self):
        self.model.train()
        epoch_loss = 0
        
        for i, batch in enumerate(self.dataloader.train_iterator):
            src = batch.src
            trg = batch.trg
            
            self.optimizer.zero_grad()
            output = self.model(src, trg, config.TEACHER_FORCING_RATIO)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            
            trg = trg[1:].view(-1)
            loss = self.criterion(output, trg)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP_SIZE)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.dataloader.train_iterator)
        
        
    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader.valid_iterator):
                src = batch.src
                trg = batch.trg
                
                output = self.model(src, trg, teacher_forching_ratio=0)
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = self.criterion(output, trg)
                
                epoch_loss += loss.item()
                
        return epoch_loss / len(self.dataloader.valid_iterator)
                
        
    
        
if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()