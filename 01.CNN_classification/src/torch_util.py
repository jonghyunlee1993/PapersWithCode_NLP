import time
import torch
import numpy as np


def train(model, iterator, optimizer, criterion, args):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        if args.max_norm_scaling:
            max_val = 3
            eps     = 1e-5
            param = model.fc.weight.norm()
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param.data = param * (desired / (eps + norm))

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def max_norm_scaling(model, max_val=3, eps=1e-8):
    param = model.fc.weight.norm()
    norm = param.norm(2, dim=0, keepdim=True)
    desired = torch.clamp(norm, 0, max_val)
    param.data = param * (desired / (eps + norm))

    return param.data


def proc_special_token(embedding, TEXT, EMBEDDING_DIM):
    UNK_INDEX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_INDEX = TEXT.vocab.stoi[TEXT.pad_token]

    embedding.weight.data[UNK_INDEX] = torch.nn.init.uniform_(torch.empty(EMBEDDING_DIM), -0.25, 0.25)
    embedding.weight.data[PAD_INDEX] = torch.zeros(EMBEDDING_DIM)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def save_model_param(fname, model):
    torch.save(model.state_dict(), fname)
    print("\nbest model parameter was saved!")


def print_training_log(epoch, start_time, end_time, train_loss, train_acc, valid_loss, valid_acc):
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print()
    print(f'\tEpoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\tVal Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def print_evaluation_log(test_loss, test_acc):
    print()
    print(f'\tVal Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}%')


def predict(TEXT, args, model, fixed_length=56):
    word2index = []

    for word in args.input_sent.split():
        word2index.append(TEXT.vocab.stoi[word])

    word2index = word2index + [1] * (fixed_length - len(word2index))
    input_tensor = torch.LongTensor(word2index).to(args.device).unsqueeze(0)
    probability = np.squeeze(torch.sigmoid(model(input_tensor)).detach().numpy()[0], 0)
    predicted_label = 'Positive' if probability >= 0.5 else 'Negative'

    return probability, predicted_label


def print_predict_log(args, probability, predicted_label):
    print()
    print(f"\tinput sent: {args.input_sent}")
    print(f'\tPredicted Label: {predicted_label} |  Probability: {probability * 100:.2f}%')


def get_time():
    return time.time()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    accuracy = correct.sum() / len(correct)

    return accuracy
