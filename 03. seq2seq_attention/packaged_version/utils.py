import math
import torch
from sacrebleu import corpus_bleu


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    print(f'Elapsed time: {elapsed_mins}m {elapsed_secs}s')
    

def print_loss(train_loss, valid_loss):
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    

def get_bleu_score(output, trg, trg_field):
    with torch.no_grad():
        output_token = output.argmax(-1)

    output_token = output_token.permute(1, 0)
    trg = trg.permute(1, 0)
    system = get_itos_batch(output_token, trg_field)
    refs = get_itos_batch(trg, trg_field)
    bleu_score = corpus_bleu(system, [refs], force=True).score

    return bleu_score


def get_speical_token(field):
    def get_stoi(idx):
        return field.vocab.stoi[idx]
    return [get_stoi(field.pad_token), get_stoi(field.unk_token), 
            get_stoi(field.eos_token)]


def get_itos_str(tokens, field):
    ignore_idx = get_speical_token(field)
    return ' '.join([field.vocab.itos[token] for token in tokens
                    if token not in ignore_idx])
    
    
def get_itos_batch(tokens_batch, field):
    return [get_itos_str(batch, field) for batch in tokens_batch]


def print_bleu(train_bleu, val_bleu):
    print(f'\t Train. bleu: {train_bleu:.3f} |  Val. bleu: {val_bleu:.3f}')