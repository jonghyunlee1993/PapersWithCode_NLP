import torch
import numpy as np


def train(model, iterator, optimizer, criterion, max_norm_sacling=False):
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

        if max_norm_sacling:
            model.param.data = max_norm_sacling(model)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def max_norm_scailing(model, max_val=3, eps=1e-8):
    param = model.fc.weight.norm()
    norm = param.norm(2, dim=0, keepdim=True)
    desired = torch.clamp(norm, 0, max_val)
    param = param * (desired / (eps + norm))

    return param


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


def proc_special_token(model, UNK_IDX, PAD_IDX, EMBEDDING_DIM):
    model.embedding.weight.data[UNK_IDX] = torch.nn.init.uniform_(torch.empty(EMBEDDING_DIM), -0.25, 0.25)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    return model


def predict(TEXT, sentence, model, device, fixed_length=56):
    word2index = []

    for word in sentence.split():
        word2index.append(TEXT.vocab.stoi[word])

    word2index = word2index + [1] * (fixed_length - len(word2index))
    input_tensor = torch.LongTensor(word2index).to(device).unsqueeze(0)
    probability = np.squeeze(torch.sigmoid(model(input_tensor)).detach().numpy()[0], 0)
    predicted_label = 'Positive' if probability >= 0.5 else 'Negative'

    return probability, predicted_label


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