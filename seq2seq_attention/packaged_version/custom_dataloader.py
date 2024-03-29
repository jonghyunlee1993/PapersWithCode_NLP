import spacy
import fr_core_news_sm, en_core_web_sm
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

class CustomDataloader:
    def __init__(self, BATCH_SIZE, DEVICE):
        self.spacy_fr = fr_core_news_sm.load()
        self.spacy_en = en_core_web_sm.load()
        
        self.init_token = "<sos>"
        self.eos_token  = "<eos>"
        self.pad_token  = "<pad>"
        self.unk_token  = "<unk>"
        
        self.BATCH_SIZE = BATCH_SIZE
        self.DEVICE     = DEVICE
    
    
    def run(self):
        self.build_vocab()            
        self.generate_iterator()
        
        
    def build_vocab(self):
        def tokenize_fr(text):
            tokenized_fr = [tok.text for tok in spacy_fr.tokenizer(text)]
            return tokenized_fr

        def tokenize_en(text):
            tokenized_en = [tok.text for tok in spacy_en.tokenizer(text)]
            return tokenized_en
        
        spacy_fr = self.spacy_fr
        spacy_en = self.spacy_en
        
        self.SRC = Field(tokenize=tokenize_fr,
                         init_token=self.init_token,
                         eos_token=self.eos_token,
                         pad_token=self.pad_token,
                         unk_token=self.unk_token,
                         lower=True)
        
        self.TRG = Field(tokenize=tokenize_en,
                         init_token=self.init_token,
                         eos_token=self.eos_token,
                         pad_token=self.pad_token,
                         unk_token=self.unk_token,
                         lower=True)
        
        self.train_data, self.valid_data, self.test_data = Multi30k.splits(exts=(".fr", ".en"), fields=(self.SRC, self.TRG))

        self.SRC.build_vocab(self.train_data, min_freq=2)
        self.TRG.build_vocab(self.train_data, min_freq=2)
        
        # print(self.SRC.unk_token)
        
        
    def generate_iterator(self):
        self.train_iterator, self.valid_iterator, self.test_iterrator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data), 
            batch_size=self.BATCH_SIZE, 
            device=self.DEVICE,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src)
        )


if __name__ == "__main__":
	device = "cpu"
	data_loader = CustomDataloader(128, device)
	data_loader.run()

